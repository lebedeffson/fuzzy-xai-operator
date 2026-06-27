from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from apps.chapter5_web_demo import DemoBackend, build_backend, evaluate_vector
from fuzzyxai.datasets import get_dataset_mode, load_dataset_mode


@dataclass
class LayeredCaseService:
    backend: DemoBackend

    @classmethod
    def create(cls) -> 'LayeredCaseService':
        return cls(backend=build_backend())

    def scenario_indices(self) -> dict[str, int]:
        proba = self.backend.model.predict_proba(self.backend.x_test)[:, 0]
        return {
            'safe': int(np.argmin(proba)),
            'risky': int(np.argmax(proba)),
            'ambiguous': int(np.argmin(np.abs(proba - 0.5))),
        }

    def vector_from_scenario(
        self,
        scenario: str,
        *,
        sample_index: int | None = None,
        manual_vector: np.ndarray | None = None,
    ) -> tuple[np.ndarray, str, int | str]:
        if scenario == 'manual' and manual_vector is not None:
            return manual_vector.astype(float), 'manual', '-'
        if sample_index is not None:
            idx = int(max(0, min(sample_index, len(self.backend.x_test) - 1)))
        else:
            idx_map = self.scenario_indices()
            if scenario in {'rupture', 'context_block'}:
                idx = idx_map['ambiguous']
            else:
                idx = idx_map.get(scenario, 0)
        vec = self.backend.x_test.iloc[idx].to_numpy(dtype=float)
        true_y = int(self.backend.y_test.iloc[idx])
        return vec, f'sample_{idx}', true_y


@dataclass
class DatasetCase:
    dataset_name: str
    domain: str
    task_type: str
    features: dict[str, float]
    target: int | float | str | None
    prediction: int | float | str | None
    predicted_risk: float
    uncertainty: float
    explain_plan: dict[str, Any]
    scenario_tags: list[str]
    purpose: str
    source: str
    status: str  # READY / MISSING / ERROR


def _build_uncertainty_profile(scenario: str, out: dict[str, Any]) -> dict[str, bool]:
    return {
        'u_num': True,
        'u_int': float(out.get('uncertainty', 0.0)) >= 0.30,
        'u_expert': scenario in {'ambiguous', 'risky', 'rupture', 'context_block'},
        'u_conf': bool(out.get('rupture', False)) or scenario in {'rupture', 'context_block'},
        'u_trace': True,
    }


def _coverage_for(cls_name: str, profile: dict[str, bool]) -> int:
    required = {k for k, v in profile.items() if v}
    supports = {
        'F0': {'u_num', 'u_trace'},
        'F_int': {'u_num', 'u_int', 'u_trace'},
        'F_H': {'u_num', 'u_expert', 'u_trace'},
        'F_N_src': {'u_num', 'u_expert', 'u_conf', 'u_trace'},
        'FML-audit': {'u_num', 'u_int', 'u_expert', 'u_conf', 'u_trace'},
    }[cls_name]
    return len(required & supports)


def _build_hierarchy_rows(profile: dict[str, bool], out: dict[str, Any]) -> tuple[list[dict[str, Any]], str, float]:
    required_n = sum(1 for v in profile.values() if v)
    rows: list[dict[str, Any]] = []
    delta_base = float(out.get('risk_breakdown', {}).get('reduction_loss', 0.0))
    class_meta = [
        ('F0', 'low', 0.45 + delta_base),
        ('F_int', 'low', 0.31 + delta_base * 0.8),
        ('F_H', 'medium', 0.28 + delta_base * 0.6),
        ('F_N_src', 'medium', 0.16 + delta_base * 0.4),
        ('FML-audit', 'high', max(0.02, 0.07 + delta_base * 0.2)),
    ]
    selected = 'F0'
    best_score = -1.0
    best_delta = 1.0
    for cls_name, complexity, delta in class_meta:
        cov = _coverage_for(cls_name, profile)
        status = 'candidate'
        if cov < required_n:
            status = 'rejected'
        score = cov / max(1, required_n) - delta * 0.05
        if cov == required_n and score > best_score:
            best_score = score
            selected = cls_name
            best_delta = delta
        rows.append(
            {
                'class': cls_name,
                'coverage': f'{cov}/{required_n}',
                'complexity': complexity,
                'reduction_loss': round(float(delta), 4),
                'status': status,
            }
        )
    if selected == 'F0':
        # if no full coverage, pick maximal coverage minimal delta
        cand = sorted(rows, key=lambda r: (-int(str(r['coverage']).split('/')[0]), float(r['reduction_loss'])))
        selected = cand[0]['class']
        best_delta = float(cand[0]['reduction_loss'])
    for row in rows:
        if row['class'] == selected:
            row['status'] = 'selected'
    return rows, selected, float(best_delta)


def _dataset_case(
    dataset_mode: str,
    *,
    vec: np.ndarray,
    true_y: int | float | str | None,
    prediction: int | float | str | None,
    predicted_risk: float,
    uncertainty: float,
    scenario: str,
) -> DatasetCase:
    spec = get_dataset_mode(dataset_mode)
    status = 'READY'
    source = 'builtin'
    task_type = 'binary_classification'
    try:
        record, _df = load_dataset_mode(dataset_mode)
        source = str(record.source)
        task_type = str(record.task_type or task_type)
    except FileNotFoundError:
        status = 'MISSING'
    except Exception:
        status = 'ERROR'

    return DatasetCase(
        dataset_name=dataset_mode,
        domain=str(spec.domain),
        task_type=task_type,
        features={f'f_{i}': float(v) for i, v in enumerate(vec)},
        target=true_y,
        prediction=prediction,
        predicted_risk=float(predicted_risk),
        uncertainty=float(uncertainty),
        explain_plan={'gamma_max': 0.45, 'thresholds': [0.10, 0.25, 0.50, 0.75]},
        scenario_tags=[scenario],
        purpose=str(spec.purpose),
        source=source,
        status=status,
    )


def build_case_state(
    service: LayeredCaseService,
    scenario: str,
    *,
    sample_index: int | None = None,
    manual_vector: np.ndarray | None = None,
    dataset_mode: str = 'breast_cancer',
) -> dict[str, Any]:
    vec, sample_id, true_y = service.vector_from_scenario(scenario, sample_index=sample_index, manual_vector=manual_vector)
    out = evaluate_vector(service.backend, vec, sample_id=sample_id)

    forced_rupture = scenario in {'rupture', 'context_block'} and not bool(out.get('rupture', False))
    if forced_rupture:
        out['rupture'] = True
        out['diagnostics'] = list(out.get('diagnostics', [])) + ['D_ij_forced_demo']
        rb = dict(out.get('risk_breakdown', {}))
        rb['diagnostic'] = 1.0
        out['risk_breakdown'] = rb
        out['action'] = 'block'
        out['rho'] = max(0.75, float(out.get('rho', 0.0)))

    profile = _build_uncertainty_profile(scenario, out)
    hierarchy_rows, selected_class, delta = _build_hierarchy_rows(profile, out)

    gamma_mr = out.get('gamma_model_risk')
    gamma_ra = out.get('gamma_risk_action')
    _gamma_max = float(out.get('thresholds', [0, 0, 0, 0])[-1])
    chi_r = 1 if list(out.get('diagnostics', [])) else 0
    chi_r_crit = 1 if bool(out.get('rupture', False)) else 0

    edges = [
        {
            'source': 'E_model',
            'target': 'E_risk',
            'gamma': gamma_mr,
            'gamma_max': 0.45,
            'status': 'morphism' if gamma_mr is not None and not out['rupture'] else 'rupture',
            'hott': 'Path exists' if gamma_mr is not None and not out['rupture'] else 'Rupture inhabited',
        },
        {
            'source': 'E_risk',
            'target': 'E_action',
            'gamma': gamma_ra,
            'gamma_max': 0.45,
            'status': 'morphism' if gamma_ra is not None and not out['rupture'] else 'rupture',
            'hott': 'Path exists' if gamma_ra is not None and not out['rupture'] else 'Rupture inhabited',
        },
    ]

    context_map = {row['object']: row for row in out.get('contexts', [])}
    ds_case = _dataset_case(
        dataset_mode,
        vec=vec,
        true_y=true_y,
        prediction=int(out['prediction']),
        predicted_risk=float(out['prob_malignant']),
        uncertainty=float(out['uncertainty']),
        scenario=scenario,
    )
    risk_components = dict(out.get('risk_breakdown', {}))
    risk_components['chi_R'] = chi_r
    risk_components['chi_R_crit'] = chi_r_crit

    return {
        'dataset': {
            'name': ds_case.dataset_name,
            'domain': ds_case.domain,
            'purpose': ds_case.purpose,
            'status': ds_case.status,
            'source': ds_case.source,
            'task_type': ds_case.task_type,
        },
        'dataset_case': asdict(ds_case),
        'scenario': scenario,
        'forced_rupture': forced_rupture,
        'input': {
            'sample_id': sample_id,
            'features': {k: float(v) for k, v in zip(service.backend.feature_names, vec)},
        },
        'model': {
            'prediction': int(out['prediction']),
            'true_y': true_y,
            'p_malignant': float(out['prob_malignant']),
            'uncertainty': float(out['uncertainty']),
            'predicted_risk': float(out['prob_malignant']),
        },
        'explanation': {
            'E_model': {
                'L': ['low_risk', 'high_risk'],
                'mu': {'low': round(1.0 - float(out['prob_malignant']), 4), 'high': round(float(out['prob_malignant']), 4)},
                'R': ['r_low_risk', 'r_high_risk'],
                'alpha': {'r_low_risk': round(1.0 - float(out['prob_malignant']), 4), 'r_high_risk': round(float(out['prob_malignant']), 4)},
                'u': round(float(out['uncertainty']), 4),
                'tau': {'id': sample_id, 'version': 'rf_breast_cancer_v1'},
            },
            'E_risk': {'L': ['low_risk', 'medium_risk', 'high_risk']},
            'E_action': {'L': ['accept', 'lower_confidence', 'request_more_data', 'defer_to_human', 'block']},
            'I_pre': float(out['I_pre']),
        },
        'composition': {
            'gamma_max': 0.45,
            'edges': edges,
            'ruptures': list(out.get('diagnostics', [])),
        },
        'uncertainty': {
            'profile': profile,
            'selected_class': selected_class,
            'delta': delta,
            'classes': hierarchy_rows,
        },
        'contexts': {
            'RiskContext': {
                'E_model': context_map.get('E_model', {}).get('RiskContext', ''),
                'E_risk': context_map.get('E_risk', {}).get('RiskContext', ''),
                'E_action': context_map.get('E_action', {}).get('RiskContext', ''),
            },
            'AutoAccept': {
                'E_model': context_map.get('E_model', {}).get('AutoAccept', ''),
                'E_risk': context_map.get('E_risk', {}).get('AutoAccept', ''),
                'E_action': context_map.get('E_action', {}).get('AutoAccept', ''),
            },
        },
        'risk': {
            'rho': float(out['rho']),
            'components': risk_components,
            'action': str(out['action']),
            'rupture': bool(out['rupture']),
            'chi_R': chi_r,
            'chi_R_crit': chi_r_crit,
            'reason': 'critical rupture' if out['rupture'] else 'threshold policy',
            'thresholds': tuple(float(v) for v in out.get('thresholds', (0.1, 0.25, 0.5, 0.75))),
            'weights': dict(out.get('risk_weights', {})),
        },
        'route_header': {
            'Input': 'built',
            'Model': 'built',
            'Omega': 'built',
            'Expl': 'rupture' if out['rupture'] else 'morphism',
            'Fuzzy': f"selected:{selected_class}",
            'Topos': 'restricted',
            'Observer': 'blocked' if out['action'] == 'block' else 'decided',
            'Action': out['action'],
        },
    }


def export_case_state(case_state: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(case_state, ensure_ascii=False, indent=2), encoding='utf-8')
