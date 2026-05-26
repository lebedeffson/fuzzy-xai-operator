from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from fuzzyxai.adapters import (
    MedicalImageToExplanationAdapter,
    TabularToExplanationAdapter,
    TextToExplanationAdapter,
)
from fuzzyxai.category import (
    AuditContext,
    ExplanationCategory,
    PresheafToposDescriptor,
    RiskContext,
    auto_accept_subpresheaf,
    try_make_morphism,
)
from fuzzyxai.core.composition import compose
from fuzzyxai.core.diagnostics import DiagnosticState
from fuzzyxai.core.explain_plan import ExplainPlan
from fuzzyxai.core.explanation_object import ExplanationObject, Rule, Trace
from fuzzyxai.data import (
    DatasetRecord,
    guess_target_column,
    load_citr_dataset,
    load_rikord_dataset,
    load_ruccod_dataset,
    split_features_target,
)
from fuzzyxai.hierarchy.f0 import F0
from fuzzyxai.hott import ExplanationPath, RuptureType
from fuzzyxai.risk import RiskAwareObserver, compute_application_risk
from fuzzyxai.trust import compute_interpretability_index


def _num(x: Any, n: int = 4) -> str:
    try:
        return f'{float(x):.{n}f}'
    except Exception:
        return '-'


def _breast_cancer_bundle() -> tuple[DatasetRecord, pd.DataFrame]:
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.copy()
    df['risk_target'] = (df['target'] == 0).astype(int)
    df = df.drop(columns=['target'])
    rec = DatasetRecord(
        name='sklearn_breast_cancer',
        source='sklearn.datasets',
        target_column='risk_target',
        task_type='binary_classification',
        description='Breast cancer baseline.',
        metadata={'fallback': False},
    )
    return rec, df


def _load_dataset_bundle(dataset: str, allow_fallback: bool) -> tuple[DatasetRecord, pd.DataFrame, str, dict[str, Any]]:
    if dataset == 'breast_cancer':
        rec, df = _breast_cancer_bundle()
    elif dataset == 'citr':
        rec, df = load_citr_dataset(mode='registry_mosmed_doctor_analysis', allow_fallback=allow_fallback)
    elif dataset == 'rikord':
        rec, df = load_rikord_dataset(allow_fallback=allow_fallback)
    elif dataset == 'ruccod':
        rec, df = load_ruccod_dataset(allow_fallback=allow_fallback)
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')

    target = rec.target_column or guess_target_column(df)
    if target is None:
        raise ValueError(f'Target column is not found for dataset={dataset}')
    metadata = {
        'requested_dataset': dataset,
        'resolved_dataset': rec.name,
        'source': rec.source,
        'fallback': bool(rec.metadata.get('fallback', False)),
        'fallback_reason': rec.metadata.get('fallback_reason'),
    }
    return rec, df, str(target), metadata


def _risk_explanation(sample_id: str, predicted_risk: float, uncertainty: float) -> ExplanationObject:
    p = max(0.0, min(1.0, float(predicted_risk)))
    return ExplanationObject(
        terms={'low_risk', 'medium_risk', 'high_risk'},
        representation=F0(lambda _x, val=p: val, label='risk_module'),
        rules=[
            Rule('risk_low', {'predicted_risk': 'low'}, 'accept'),
            Rule('risk_medium', {'predicted_risk': 'medium'}, 'lower_confidence'),
            Rule('risk_high', {'predicted_risk': 'high'}, 'defer_to_human'),
        ],
        activations={
            'risk_low': float(max(0.0, 1.0 - p * 1.4)),
            'risk_medium': float(max(0.0, 1.0 - abs(p - 0.5) * 4.0)),
            'risk_high': float(p),
        },
        uncertainty=float(uncertainty),
        trace=Trace(
            id=f'{sample_id}:risk',
            version='risk_module_v1',
            timestamp=datetime.now(timezone.utc).isoformat(),
            source='risk_module',
            checksum=f'{sample_id}:risk',
        ),
        metadata={'activation_threshold': 0.05},
    )


def _action_explanation(sample_id: str, action: str, rho: float) -> ExplanationObject:
    return ExplanationObject(
        terms={'accept', 'lower_confidence', 'request_more_data', 'defer_to_human', 'block'},
        representation=F0(lambda _x, val=rho: val, label='action_module'),
        rules=[Rule(f'action_{action}', {'rho': action}, action)],
        activations={f'action_{action}': 1.0},
        uncertainty=0.0,
        trace=Trace(
            id=f'{sample_id}:action',
            version='action_module_v1',
            timestamp=datetime.now(timezone.utc).isoformat(),
            source='observer_action',
            checksum=f'{sample_id}:action:{action}',
        ),
        metadata={'activation_threshold': 0.05},
    )


def _to_dict_path(path: ExplanationPath) -> dict[str, Any]:
    return {
        'source': path.source,
        'target': path.target,
        'gamma_total': path.gamma_total,
        'delta_total': path.delta_total,
        'trace': list(path.trace),
        'morphisms': [m.signature for m in path.morphisms],
    }


def _to_dict_rupture(rupture: RuptureType) -> dict[str, Any]:
    return {
        'source': rupture.source,
        'target': rupture.target,
        'reason': rupture.reason,
        'gamma': rupture.gamma,
        'threshold': rupture.threshold,
        'code': rupture.diagnostic_state.code,
        'severity': rupture.diagnostic_state.severity,
        'context': dict(rupture.diagnostic_state.context),
    }


def _select_model_adapter(dataset: str, x_train_raw: pd.DataFrame) -> Any:
    if dataset == 'ruccod':
        return TextToExplanationAdapter()
    if dataset == 'citr':
        lower = {str(c).lower() for c in x_train_raw.columns}
        if any(k in lower for k in {'image_path', 'dicom_path', 'file_path'}):
            return MedicalImageToExplanationAdapter()
    return TabularToExplanationAdapter().fit(x_train_raw)


def _build_model_explanation(
    dataset: str,
    adapter: Any,
    row: pd.Series,
    *,
    sample_id: str,
    predicted_risk: float,
    model_version: str,
    source: str,
) -> ExplanationObject:
    if isinstance(adapter, TextToExplanationAdapter):
        text_col = next((c for c in row.index if 'text' in str(c).lower()), row.index[0])
        icd_col = next((c for c in row.index if 'icd' in str(c).lower()), None)
        text = str(row.get(text_col, ''))
        codes = [str(row.get(icd_col, '')).strip()] if icd_col else []
        return adapter.adapt(
            sample_id=sample_id,
            text=text,
            icd_codes=codes,
            predicted_risk=predicted_risk,
            model_version=model_version,
            source=source,
        )
    if isinstance(adapter, MedicalImageToExplanationAdapter):
        image_col = next((c for c in row.index if 'path' in str(c).lower()), None)
        modality_col = next((c for c in row.index if 'modality' in str(c).lower()), None)
        protocol_col = next((c for c in row.index if 'protocol' in str(c).lower()), None)
        return adapter.adapt(
            sample_id=sample_id,
            predicted_risk=predicted_risk,
            image_path=(str(row.get(image_col)) if image_col else None),
            metadata={
                'modality': str(row.get(modality_col, '')) if modality_col else '',
                'protocol': str(row.get(protocol_col, '')) if protocol_col else '',
            },
            model_version=model_version,
            source=source,
        )
    return adapter.adapt(
        row,
        sample_id=sample_id,
        predicted_risk=predicted_risk,
        model_version=model_version,
        source=source,
    )


def _distribution_and_diagnostics(
    dataset: str,
    adapter: Any,
    x_test_raw: pd.DataFrame,
    proba_rows: list[list[float]],
    *,
    beta: dict[str, float],
    lambda_weights: dict[str, float],
    max_samples: int = 256,
) -> tuple[list[float], list[dict[str, Any]]]:
    i_values: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    lim = min(len(x_test_raw), max_samples)
    for i in range(lim):
        row = x_test_raw.iloc[i]
        probs = proba_rows[i]
        p_risk = float(probs[1]) if len(probs) > 1 else float(max(probs))
        sample_id = f'dist_{i}'
        e_model = _build_model_explanation(
            dataset,
            adapter,
            row,
            sample_id=sample_id,
            predicted_risk=p_risk,
            model_version='rf_dataset_v1',
            source=f'{dataset}_distribution',
        )
        i_pre = float(compute_interpretability_index(e_model, lambda_weights=lambda_weights, lambda_delta=0.10))
        i_values.append(i_pre)
        uncertainty = float(1.0 - abs((probs[1] if len(probs) > 1 else probs[0]) - probs[0])) if len(probs) > 1 else 0.5
        e_risk = _risk_explanation(sample_id, p_risk, uncertainty)
        comp = compose(e_model, e_risk, beta, allow_missing_terms=True)
        if isinstance(comp, DiagnosticState):
            diagnostics.append(
                {
                    'sample_id': sample_id,
                    'code': comp.code,
                    'reason': comp.reason,
                    'severity': comp.severity,
                    'context': dict(comp.context),
                }
            )
    return i_values, diagnostics


def run(
    *,
    dataset: str = 'breast_cancer',
    seed: int = 42,
    sample_index: int = 0,
    allow_fallback: bool = True,
) -> tuple[dict[str, Any], list[float], list[dict[str, Any]]]:
    record, df, target_column, dataset_metadata = _load_dataset_bundle(dataset, allow_fallback)
    x_raw, y_raw = split_features_target(df, target_column)

    y_encoder = LabelEncoder()
    y = pd.Series(y_encoder.fit_transform(y_raw), index=y_raw.index, name=target_column)
    if y.nunique() < 2:
        raise ValueError('Target must contain at least 2 classes')

    x_model = pd.get_dummies(x_raw, dummy_na=True)
    idx_all = y.index.to_list()
    stratify = y if y.value_counts().min() >= 2 and y.nunique() <= 20 else None
    idx_train, idx_test = train_test_split(
        idx_all,
        test_size=0.25,
        random_state=seed,
        stratify=stratify,
    )
    x_train = x_model.loc[idx_train]
    x_test = x_model.loc[idx_test]
    y_train = y.loc[idx_train]
    y_test = y.loc[idx_test]
    x_train_raw = x_raw.loc[idx_train]
    x_test_raw = x_raw.loc[idx_test]

    model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=seed)
    model.fit(x_train, y_train)
    proba_test = model.predict_proba(x_test)
    pred_test = model.predict(x_test)

    pos_idx = 1 if proba_test.shape[1] > 1 else 0
    idx = max(0, min(int(sample_index), len(x_test_raw) - 1))
    row = x_test_raw.iloc[idx]
    true_y = int(y_test.iloc[idx])
    pred_y = int(pred_test[idx])
    probs = [float(v) for v in proba_test[idx].tolist()]
    p_risk = float(probs[pos_idx]) if len(probs) > pos_idx else float(max(probs))
    uncertainty = float(1.0 - abs((probs[pos_idx] if len(probs) > 1 else probs[0]) - probs[0])) if len(probs) > 1 else 0.5

    plan = ExplainPlan.from_data(x_train, y_train, mode='audit').with_reduction_weight(0.10)
    beta = dict(plan.beta)
    beta['gamma_max'] = 0.45
    observer = RiskAwareObserver(plan=plan)

    adapter = _select_model_adapter(dataset, x_train_raw)
    sample_id = f'{dataset}_case_{idx}'
    e_model = _build_model_explanation(
        dataset,
        adapter,
        row,
        sample_id=sample_id,
        predicted_risk=p_risk,
        model_version='rf_dataset_v1',
        source=record.name,
    )
    i_pre = float(compute_interpretability_index(e_model, lambda_weights=plan.lambda_, lambda_delta=0.10))
    e_risk = _risk_explanation(sample_id, p_risk, uncertainty)

    cat = ExplanationCategory(beta=beta)
    o_model = cat.object('E_model', e_model)
    o_risk = cat.object('E_risk', e_risk)
    mr_result = try_make_morphism(cat, o_model, o_risk, name='T_model_risk')

    path_model_risk: ExplanationPath | None = None
    rupture_model_risk: RuptureType | None = None
    if mr_result.success and mr_result.morphism is not None:
        path_model_risk = ExplanationPath.from_morphism(mr_result.morphism)
    else:
        rupture_model_risk = RuptureType.from_diagnostic(
            'E_model',
            'E_risk',
            mr_result.diagnostic or DiagnosticState('D_ij', 'unknown category error', 'critical'),
            gamma=(mr_result.diagnostic.context.get('gamma') if mr_result.diagnostic else None),
            threshold=cat.gamma_max,
        )

    diagnostics_codes: list[str] = []
    comp_mr = compose(e_model, e_risk, beta, allow_missing_terms=True)
    if isinstance(comp_mr, DiagnosticState):
        diagnostics_codes.append(comp_mr.code)

    risk_breakdown = compute_application_risk(
        predicted_risk=p_risk,
        uncertainty=uncertainty,
        pre_interpretability=i_pre,
        reduction_loss=float(e_model.reduction_loss),
        diagnostics=diagnostics_codes,
        weights=observer.weights,
    )
    decision = observer.evaluate(
        predicted_risk=p_risk,
        uncertainty=uncertainty,
        pre_interpretability=i_pre,
        reduction_loss=float(e_model.reduction_loss),
        diagnostics=diagnostics_codes,
    )

    e_action = _action_explanation(sample_id, decision.action, decision.rho)
    o_action = cat.object('E_action', e_action)
    ra_result = try_make_morphism(cat, o_risk, o_action, name='T_risk_action')

    path_risk_action: ExplanationPath | None = None
    rupture_risk_action: RuptureType | None = None
    if ra_result.success and ra_result.morphism is not None:
        path_risk_action = ExplanationPath.from_morphism(ra_result.morphism)
    else:
        rupture_risk_action = RuptureType.from_diagnostic(
            'E_risk',
            'E_action',
            ra_result.diagnostic or DiagnosticState('D_ij', 'unknown category error', 'critical'),
            gamma=(ra_result.diagnostic.context.get('gamma') if ra_result.diagnostic else None),
            threshold=cat.gamma_max,
        )

    full_path: ExplanationPath | None = None
    if path_model_risk and path_risk_action:
        full_path = path_model_risk.concat(path_risk_action)

    risk_ctx = RiskContext(
        cat,
        {
            o_model: {'accept', 'lower_confidence', 'request_more_data', 'defer_to_human', 'block'},
            o_risk: {'lower_confidence', 'request_more_data', 'defer_to_human', 'block'},
            o_action: {decision.action},
        },
    )
    audit_ctx = AuditContext(
        cat,
        {
            o_model: {'trace_id', 'trace_version', 'checksum'},
            o_risk: {'trace_id', 'trace_version'},
            o_action: {'trace_id'},
        },
    )
    auto_accept = auto_accept_subpresheaf(risk_ctx)
    restricted_context: list[str] = []
    if mr_result.success and mr_result.morphism is not None:
        source_actions = risk_ctx.allowed_actions(o_model)

        def restrict_to_model(action: str) -> str:
            return action if action in source_actions else 'block'

        risk_ctx.set_restriction(mr_result.morphism, restrict_to_model)
        restricted_context = sorted({risk_ctx.restrict(mr_result.morphism, a) for a in risk_ctx.allowed_actions(o_risk)})

    descriptor = PresheafToposDescriptor()
    i_values, diagnostics_examples = _distribution_and_diagnostics(
        dataset,
        adapter,
        x_test_raw.reset_index(drop=True),
        [list(map(float, row)) for row in proba_test.tolist()],
        beta=beta,
        lambda_weights=plan.lambda_,
    )

    report = {
        'meta': {
            'dataset': dataset,
            'resolved_dataset': record.name,
            'seed': seed,
            'sample_index': idx,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'route': 'data -> model -> E_model -> Expl/HoTT/topos -> rho -> action',
        },
        'dataset_metadata': {
            **dataset_metadata,
            'dataset_record': record.as_trace(),
            'n_rows': int(len(df)),
            'n_features': int(x_raw.shape[1]),
            'target_column': target_column,
        },
        'model': {
            'true_y': true_y,
            'pred_y': pred_y,
            'probabilities': probs,
            'predicted_risk': p_risk,
            'uncertainty': uncertainty,
        },
        'interpretability': {
            'I_pre': i_pre,
            'reduction_loss': float(e_model.reduction_loss),
        },
        'risk': {
            'weights': dict(observer.weights),
            'thresholds': list(observer.thresholds),
            'breakdown': risk_breakdown.as_dict(),
            'rho': float(decision.rho),
            'action': decision.action,
            'reason': decision.reason,
        },
        'category': {
            'gamma_max': cat.gamma_max,
            'objects': [o.key for o in cat.objects()],
            'morphisms': [m.signature for m in cat.morphisms()],
            'model_to_risk_success': mr_result.success,
            'risk_to_action_success': ra_result.success,
        },
        'hott': {
            'path_model_risk': _to_dict_path(path_model_risk) if path_model_risk else None,
            'path_risk_action': _to_dict_path(path_risk_action) if path_risk_action else None,
            'path_full': _to_dict_path(full_path) if full_path else None,
            'rupture_model_risk': _to_dict_rupture(rupture_model_risk) if rupture_model_risk else None,
            'rupture_risk_action': _to_dict_rupture(rupture_risk_action) if rupture_risk_action else None,
        },
        'topos': {
            'descriptor': asdict(descriptor),
            'risk_context': {
                o_model.key: sorted(risk_ctx.allowed_actions(o_model)),
                o_risk.key: sorted(risk_ctx.allowed_actions(o_risk)),
                o_action.key: sorted(risk_ctx.allowed_actions(o_action)),
            },
            'auto_accept': {
                o_model.key: sorted(auto_accept(o_model)),
                o_risk.key: sorted(auto_accept(o_risk)),
                o_action.key: sorted(auto_accept(o_action)),
            },
            'audit_context': {
                o_model.key: sorted(audit_ctx(o_model)),
                o_risk.key: sorted(audit_ctx(o_risk)),
                o_action.key: sorted(audit_ctx(o_action)),
            },
            'restricted_context_model_from_risk': restricted_context,
        },
    }
    return report, i_values, diagnostics_examples


def write_real_data_artifacts(
    dataset: str,
    report: dict[str, Any],
    i_values: list[float],
    diagnostics_examples: list[dict[str, Any]],
    out_dir: str | Path,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}

    json_path = out / f'{dataset}_report.json'
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    result['json'] = json_path

    i_csv = out / f'{dataset}_i_pre_distribution.csv'
    pd.DataFrame({'I_pre': i_values}).to_csv(i_csv, index=False)
    result['i_pre_csv'] = i_csv

    diag_path = out / f'{dataset}_diagnostics_examples.json'
    diag_path.write_text(json.dumps(diagnostics_examples, ensure_ascii=False, indent=2), encoding='utf-8')
    result['diagnostics_json'] = diag_path

    thresholds_csv = out / f'{dataset}_calibrated_thresholds.csv'
    th = list(report['risk']['thresholds'])
    pd.DataFrame([{'theta_1': th[0], 'theta_2': th[1], 'theta_3': th[2], 'theta_4': th[3]}]).to_csv(thresholds_csv, index=False)
    result['thresholds_csv'] = thresholds_csv

    summary_csv = out / 'summary.csv'
    row = {
        'dataset': dataset,
        'resolved_dataset': report['dataset_metadata']['resolved_dataset'],
        'fallback': report['dataset_metadata']['fallback'],
        'predicted_risk': report['model']['predicted_risk'],
        'I_pre': report['interpretability']['I_pre'],
        'rho': report['risk']['rho'],
        'action': report['risk']['action'],
        'diagnostics_count': len(diagnostics_examples),
    }
    if summary_csv.exists():
        prev = pd.read_csv(summary_csv)
        prev = prev[prev['dataset'] != dataset]
        pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_csv(summary_csv, index=False)
    else:
        pd.DataFrame([row]).to_csv(summary_csv, index=False)
    result['summary_csv'] = summary_csv

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6, 4))
        plt.hist(i_values, bins=20)
        plt.title(f'I_pre distribution: {dataset}')
        plt.xlabel('I_pre')
        plt.ylabel('count')
        png_path = out / f'{dataset}_i_pre_distribution.png'
        fig.tight_layout()
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        result['i_pre_png'] = png_path
    except Exception:
        pass

    return result


def print_summary(report: dict[str, Any]) -> None:
    m = report['model']
    r = report['risk']
    c = report['category']
    h = report['hott']
    d = report['dataset_metadata']
    print('Unified Full Demo')
    print('-' * 68)
    print(f"dataset={d['requested_dataset']} resolved={d['resolved_dataset']} fallback={d['fallback']}")
    print(f"true={m['true_y']} pred={m['pred_y']} risk={_num(m['predicted_risk'])} uncertainty={_num(m['uncertainty'])}")
    print(f"I_pre={_num(report['interpretability']['I_pre'])} rho={_num(r['rho'])} action={r['action']}")
    print(f"Expl objects={c['objects']}")
    print(f"morphisms: model->risk={c['model_to_risk_success']} risk->action={c['risk_to_action_success']}")
    print('HoTT path full exists=', bool(h['path_full']))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['breast_cancer', 'citr', 'rikord', 'ruccod'], default='breast_cancer')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--out-dir', default='reports/chapter5/real_data_validation')
    parser.add_argument('--strict-dataset-files', action='store_true', help='do not fallback when local dataset file is missing')
    args = parser.parse_args()

    report, i_values, diagnostics_examples = run(
        dataset=args.dataset,
        seed=args.seed,
        sample_index=args.sample_index,
        allow_fallback=not args.strict_dataset_files,
    )
    paths = write_real_data_artifacts(args.dataset, report, i_values, diagnostics_examples, args.out_dir)
    print_summary(report)
    print('\nArtifacts:')
    for k, v in paths.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()

