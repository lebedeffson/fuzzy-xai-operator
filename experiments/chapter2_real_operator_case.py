from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from apps.chapter5_web_demo import build_backend, evaluate_vector
from fuzzyxai.data.breast_cancer_adapter import build_explanation_for_prediction
from fuzzyxai.trust import compute_interpretability_index


def _select_sample_index(backend, target_probability: float = 0.72) -> int:
    proba = backend.model.predict_proba(backend.x_test)[:, 0]
    return int(np.argmin(np.abs(proba - float(target_probability))))


def _top_features(backend, idx: int, n: int = 5) -> list[str]:
    x_train = backend.x_test
    feature_std = x_train.std(axis=0).replace(0.0, 1.0)
    z = (backend.x_test.iloc[idx] - x_train.mean(axis=0)).abs() / feature_std
    return z.sort_values(ascending=False).head(n).index.tolist()


def generate_case(*, out_dir: str | Path = 'reports/chapter2_real_operator_case', target_probability: float = 0.72) -> dict[str, Any]:
    backend = build_backend()
    idx = _select_sample_index(backend, target_probability=target_probability)
    sample_id = f'sample_{idx}'
    vec = backend.x_test.iloc[idx].to_numpy(dtype=float)

    out = evaluate_vector(backend, vec, sample_id=sample_id)
    exp_obj = build_explanation_for_prediction(
        float(out['prob_malignant']),
        sample_id=sample_id,
        model_version='rf_breast_cancer_v1',
        dataset_id='sklearn_breast_cancer',
    )
    activations = {k: float(v) for k, v in exp_obj.activations.items()}
    mu_low = float(activations.get('r_low_risk', 0.0))
    mu_high = float(activations.get('r_high_risk', 0.0))
    mu_medium = float(max(0.0, 1.0 - abs(mu_high - mu_low) * 2.0))
    u_rules = 1.0 - max(activations.values() or [0.0])
    u_model = float(out['uncertainty'])
    u_trace = 0.02
    u_m = float(np.clip(0.4 * u_model + 0.4 * u_rules + 0.2 * u_trace, 0.0, 1.0))
    i_pre = float(compute_interpretability_index(exp_obj, lambda_weights=backend.plan.lambda_, lambda_delta=0.10))

    result = {
        'sample_id': sample_id,
        'selected_features': _top_features(backend, idx),
        'P(malignant)': float(out['prob_malignant']),
        'p_malignant': float(out['prob_malignant']),
        'mu_low': mu_low,
        'mu_medium': mu_medium,
        'mu_high': mu_high,
        'active_rules': sorted(list(exp_obj.active_rules)),
        'alpha_values': activations,
        'U_model': u_model,
        'U_rules': u_rules,
        'U_trace': u_trace,
        'u_M': u_m,
        'tau_fields': {
            'id': exp_obj.trace.id,
            'version': exp_obj.trace.version,
            'timestamp': exp_obj.trace.timestamp,
            'source': exp_obj.trace.source,
            'checksum': exp_obj.trace.checksum,
        },
        'I_pre': i_pre,
        'gamma_model_to_risk': out.get('gamma_model_risk'),
        'chi_R': int(1 if out.get('diagnostics') else 0),
        'chi_R_crit': int(1 if out.get('rupture') else 0),
        'action': str(out.get('action')),
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / 'breast_cancer_operator_case.json'
    md_path = out_path / 'breast_cancer_operator_case.md'
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    md = [
        '# Chapter 2 Real Operator Case (Breast Cancer)',
        '',
        f"- sample_id: `{result['sample_id']}`",
        f"- selected_features: `{', '.join(result['selected_features'])}`",
        f"- P(malignant): `{result['P(malignant)']:.6f}`",
        f"- mu_low: `{result['mu_low']:.6f}`",
        f"- mu_medium: `{result['mu_medium']:.6f}`",
        f"- mu_high: `{result['mu_high']:.6f}`",
        f"- active_rules: `{result['active_rules']}`",
        f"- alpha_values: `{result['alpha_values']}`",
        f"- U_model: `{result['U_model']:.6f}`",
        f"- U_rules: `{result['U_rules']:.6f}`",
        f"- U_trace: `{result['U_trace']:.6f}`",
        f"- u_M: `{result['u_M']:.6f}`",
        f"- tau_fields: `{result['tau_fields']}`",
        f"- I_pre: `{result['I_pre']:.6f}`",
        f"- gamma_model_to_risk: `{result['gamma_model_to_risk']}`",
        f"- chi_R: `{result['chi_R']}`",
        f"- chi_R_crit: `{result['chi_R_crit']}`",
        f"- action: `{result['action']}`",
    ]
    md_path.write_text('\n'.join(md), encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='reports/chapter2_real_operator_case')
    parser.add_argument('--target-probability', type=float, default=0.72)
    args = parser.parse_args()
    result = generate_case(out_dir=args.out_dir, target_probability=args.target_probability)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
