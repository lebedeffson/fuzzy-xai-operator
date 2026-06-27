from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    import shap
except Exception:  # pragma: no cover
    shap = None
try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:  # pragma: no cover
    LimeTabularExplainer = None
try:
    from anchor import anchor_tabular
except Exception:  # pragma: no cover
    anchor_tabular = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from fuzzyxai import ExplainPlan
from fuzzyxai.data.dataset_loader import split_features_target
from fuzzyxai.data.profile_inference import infer_dataset_profile
from fuzzyxai.datasets import load_dataset_mode
from fuzzyxai.risk import RiskAwareModel, proxy_action_label, reference_action_label
from fuzzyxai.risk.representation_selection import profile_from_dataset_profile

warnings.filterwarnings(
    'ignore',
    message='X does not have valid feature names',
    category=UserWarning,
)


AUTO_ACTIONS = {'accept', 'lower_confidence'}
MAX_EXPLAIN_ROWS = 40


def _rupture_proxy_flags(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    any_r = pd.Series(False, index=df.index)
    crit_r = pd.Series(False, index=df.index)
    for col in ('chi_R', 'chi_r', 'rupture'):
        if col in df.columns:
            any_r = any_r | df[col].astype(bool)
    for col in ('chi_R_crit', 'chi_r_crit', 'critical_rupture', 'is_critical'):
        if col in df.columns:
            crit_r = crit_r | df[col].astype(bool)
    if {'expert_a', 'expert_b'}.issubset(df.columns):
        any_r = any_r | (df['expert_a'].astype(str) != df['expert_b'].astype(str))
    if {'source_model', 'source_expert'}.issubset(df.columns):
        src = df['source_model'].astype(str) != df['source_expert'].astype(str)
        any_r = any_r | src
        crit_r = crit_r | src
    any_r = any_r | crit_r
    return any_r, crit_r


def _metrics(
    actions: list[str],
    expected_proxy: list[str],
    expected_reference: list[str],
    p: np.ndarray,
    u: np.ndarray,
    i_pre: np.ndarray,
    chi_r: np.ndarray,
    chi_r_crit: np.ndarray,
) -> dict[str, Any]:
    n = max(1, len(actions))
    a = np.array(actions, dtype=object)
    exp_proxy = np.array(expected_proxy, dtype=object)
    exp_ref = np.array(expected_reference, dtype=object)
    chi_auto = ((p < 0.35) & (u < 0.45) & (i_pre >= 0.65) & (chi_r_crit == 0))
    crit_total = int(chi_r_crit.sum())
    missed_critical = int(((chi_r_crit == 1) & (a != 'block')).sum())
    return {
        'agreement_proxy': float(np.mean(a == exp_proxy)),
        'agreement_reference': float(np.mean(a == exp_ref)),
        'missed_critical_ruptures': missed_critical,
        'critical_rupture_recall': 1.0 if crit_total == 0 else float((crit_total - missed_critical) / crit_total),
        'false_auto_accept_rate': float(np.mean(np.isin(a, list(AUTO_ACTIONS)) & (~chi_auto))),
        'false_block_rate': float(np.mean((a == 'block') & (chi_r_crit == 0))),
        'auto_accept_coverage': float(np.mean(np.isin(a, list(AUTO_ACTIONS)))),
        'action_distribution': dict(Counter(actions)),
    }


def _apply_observer_policy(
    *,
    p: np.ndarray,
    u: np.ndarray,
    i_pre: np.ndarray,
    delta: np.ndarray,
    chi_r: np.ndarray,
    chi_r_crit: np.ndarray,
    trace_valid: np.ndarray,
    weights: dict[str, float],
    thresholds: tuple[float, float, float, float],
    i_min: float,
    delta_max: float,
    uncertainty_max: float,
) -> list[str]:
    t1, t2, t3, t4 = thresholds
    actions: list[str] = []
    for pp, uu, ii, dd, rr, rc, tv in zip(p, u, i_pre, delta, chi_r, chi_r_crit, trace_valid):
        chi_auto = bool((pp < 0.35) and (uu < uncertainty_max) and (ii >= i_min) and (rc == 0) and tv)
        rho = (
            weights['predicted_risk'] * float(pp)
            + weights['uncertainty'] * float(uu)
            + weights['interpretability_gap'] * float(1.0 - ii)
            + weights['reduction_loss'] * float(dd)
            + weights['diagnostic'] * float(rr)
        )
        rho = float(np.clip(rho, 0.0, 1.0))
        if int(rc) == 1:
            actions.append('block')
            continue
        if not chi_auto:
            actions.append('defer_to_human' if rho >= t3 else 'request_more_data')
            continue
        if rho < t1 and int(rr) == 0 and float(ii) >= i_min and float(dd) <= delta_max and bool(tv) and float(uu) < uncertainty_max:
            actions.append('accept')
            continue
        if rho < t2 and int(rr) == 0:
            actions.append('lower_confidence')
            continue
        if float(uu) >= uncertainty_max or float(dd) > delta_max or float(ii) < i_min or not bool(tv):
            actions.append('request_more_data')
            continue
        if rho < t4:
            actions.append('defer_to_human')
            continue
        actions.append('block')
    return actions


def _threshold_action(p: float, u: float, i_pre: float, chi_r_crit: int) -> str:
    if chi_r_crit == 1:
        return 'block'
    if p >= 0.75:
        return 'defer_to_human'
    if u >= 0.45 or i_pre < 0.65:
        return 'request_more_data'
    if p >= 0.35:
        return 'lower_confidence'
    return 'accept'


def _threshold_action_native(p: float) -> str:
    if p >= 0.80:
        return 'defer_to_human'
    if p >= 0.35:
        return 'lower_confidence'
    return 'accept'


def _shap_guardrail_action(p: float, u: float, i_pre: float, chi_r_crit: int, shap_top_abs: float, shap_q90: float) -> str:
    if chi_r_crit == 1:
        return 'block'
    if shap_top_abs >= shap_q90 and p >= 0.35:
        return 'defer_to_human'
    return _threshold_action(p, u, i_pre, chi_r_crit)


def _shap_guardrail_action_native(p: float, shap_top_abs: float, shap_q90: float) -> str:
    if shap_top_abs >= shap_q90 and p >= 0.35:
        return 'defer_to_human'
    return _threshold_action_native(p)


def _lime_guardrail_action(p: float, u: float, i_pre: float, chi_r_crit: int, lime_top_abs: float, lime_q90: float, mixed_signs: bool) -> str:
    if chi_r_crit == 1:
        return 'block'
    if mixed_signs and lime_top_abs >= lime_q90 and p >= 0.35:
        return 'defer_to_human'
    return _threshold_action(p, u, i_pre, chi_r_crit)


def _lime_guardrail_action_native(p: float, lime_top_abs: float, lime_q90: float, mixed_signs: bool) -> str:
    if mixed_signs and lime_top_abs >= lime_q90 and p >= 0.35:
        return 'defer_to_human'
    return _threshold_action_native(p)


def _anchor_guardrail_action(p: float, u: float, i_pre: float, chi_r_crit: int, precision: float, coverage: float, empty_anchor: bool) -> str:
    if chi_r_crit == 1:
        return 'block'
    weak_anchor = empty_anchor or precision < 0.95 or coverage < 0.05
    if weak_anchor and p >= 0.35:
        return 'defer_to_human'
    return _threshold_action(p, u, i_pre, chi_r_crit)


def _anchor_guardrail_action_native(p: float, precision: float, coverage: float, empty_anchor: bool) -> str:
    weak_anchor = empty_anchor or precision < 0.95 or coverage < 0.05
    if weak_anchor and p >= 0.35:
        return 'defer_to_human'
    return _threshold_action_native(p)


def run(dataset: str, *, out_root: str | Path = 'reports/datasets', baseline_access: str = 'native') -> dict[str, Any]:
    record, df = load_dataset_mode(dataset)
    x_raw, y_raw = split_features_target(df, str(record.target_column))
    excluded_metadata_columns = [c for c in x_raw.columns if str(c).lower().startswith(('expert_', 'source_'))]
    x_model = pd.get_dummies(x_raw.drop(columns=excluded_metadata_columns), dummy_na=True)
    y = pd.Series(LabelEncoder().fit_transform(y_raw), index=y_raw.index)
    stratify = y if y.value_counts().min() >= 2 and y.nunique() <= 20 else None
    x_train, x_test, y_train, _y_test = train_test_split(x_model, y, test_size=0.25, random_state=42, stratify=stratify)
    x_train_np = x_train.to_numpy()
    x_test_np = x_test.to_numpy()
    model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
    model.fit(x_train_np, y_train.to_numpy())

    plan = ExplainPlan.from_data(x_train, y_train, mode='audit').with_reduction_weight(0.10)
    profile = infer_dataset_profile(df, requires_audit=True)
    xai_profile = sorted(profile_from_dataset_profile(profile))
    proxy_any, proxy_crit = _rupture_proxy_flags(x_raw)
    test_idx = list(x_test.index)
    observer = RiskAwareModel(model, plan=plan, positive_class=1 if y.nunique() > 1 else 0)
    chi_r_flags = [int(bool(proxy_any.loc[idx])) for idx in test_idx]
    chi_r_crit_flags = [int(bool(proxy_crit.loc[idx])) for idx in test_idx]
    outs = observer.predict_with_risk(
        x_test_np,
        metadata={
            'source': record.source,
            'mode': 'audit',
            'xai_profile': xai_profile,
            'chi_r_flags': chi_r_flags,
            'chi_r_crit_flags': chi_r_crit_flags,
        },
    )
    p = np.array([float(o['predicted_risk']) for o in outs], dtype=float)
    u = np.array([float(o['uncertainty']) for o in outs], dtype=float)
    i_pre = np.array([float(o['pre_interpretability']) for o in outs], dtype=float)
    delta = np.array([float(o.get('reduction_loss', 0.0)) for o in outs], dtype=float)
    trace_valid = np.array([bool(o.get('E_model_ext', {}).get('trace')) for o in outs], dtype=bool)
    chi_r = np.array([1 if bool(outs[i].get('diagnostics')) or bool(proxy_any.loc[test_idx[i]]) else 0 for i in range(len(outs))], dtype=int)
    chi_r_crit = np.array([1 if bool(outs[i].get('diagnostics')) or bool(proxy_crit.loc[test_idx[i]]) else 0 for i in range(len(outs))], dtype=int)
    expected_proxy = [
        proxy_action_label(predicted_risk=float(pp), uncertainty=float(uu), i_pre=float(ii), rupture=bool(rr), critical_rupture=bool(rc))
        for pp, uu, ii, rr, rc in zip(p, u, i_pre, chi_r, chi_r_crit)
    ]
    y_true_test = y.loc[test_idx].to_numpy()
    y_pred_test = np.array([int(o.get('raw_prediction', o.get('prediction', 0))) for o in outs], dtype=int)
    expected_reference = [
        reference_action_label(
            predicted_risk=float(pp),
            uncertainty=float(uu),
            i_pre=float(ii),
            rupture=bool(rr),
            critical_rupture=bool(rc),
            y_true=int(yt),
            y_pred=int(yp),
        )
        for pp, uu, ii, rr, rc, yt, yp in zip(p, u, i_pre, chi_r, chi_r_crit, y_true_test, y_pred_test)
    ]

    full_actions_runtime = [str(o['action']) for o in outs]
    use_equal_guardrail = str(baseline_access) == 'equal_guardrail'
    threshold_actions = [
        _threshold_action(float(pp), float(uu), float(ii), int(rc))
        if use_equal_guardrail
        else _threshold_action_native(float(pp))
        for pp, uu, ii, rc in zip(p, u, i_pre, chi_r_crit)
    ]

    uncalibrated_params = {
        'weights': {
            'predicted_risk': 0.40,
            'uncertainty': 0.20,
            'interpretability_gap': 0.15,
            'reduction_loss': 0.10,
            'diagnostic': 0.15,
        },
        'thresholds': (0.10, 0.25, 0.50, 0.80),
        'i_min': 0.65,
        'delta_max': 0.15,
        'uncertainty_max': 0.45,
    }
    calibration_path = Path(out_root) / dataset / 'calibration.json'
    calibrated_params = dict(uncalibrated_params)
    if calibration_path.exists():
        try:
            calib = json.loads(calibration_path.read_text(encoding='utf-8'))
            bp = calib.get('best_params', {})
            calibrated_params = {
                'weights': dict(bp.get('weights', uncalibrated_params['weights'])),
                'thresholds': tuple(bp.get('thresholds', uncalibrated_params['thresholds'])),
                'i_min': float(bp.get('i_min', uncalibrated_params['i_min'])),
                'delta_max': float(bp.get('delta_max', uncalibrated_params['delta_max'])),
                'uncertainty_max': float(bp.get('uncertainty_max', uncalibrated_params['uncertainty_max'])),
            }
        except Exception:
            pass

    full_actions_uncal = _apply_observer_policy(
        p=p, u=u, i_pre=i_pre, delta=delta, chi_r=chi_r, chi_r_crit=chi_r_crit, trace_valid=trace_valid,
        weights=dict(uncalibrated_params['weights']),
        thresholds=tuple(uncalibrated_params['thresholds']),
        i_min=float(uncalibrated_params['i_min']),
        delta_max=float(uncalibrated_params['delta_max']),
        uncertainty_max=float(uncalibrated_params['uncertainty_max']),
    )
    full_actions_cal = _apply_observer_policy(
        p=p, u=u, i_pre=i_pre, delta=delta, chi_r=chi_r, chi_r_crit=chi_r_crit, trace_valid=trace_valid,
        weights=dict(calibrated_params['weights']),
        thresholds=tuple(calibrated_params['thresholds']),
        i_min=float(calibrated_params['i_min']),
        delta_max=float(calibrated_params['delta_max']),
        uncertainty_max=float(calibrated_params['uncertainty_max']),
    )

    shap_actions: list[str] = []
    if shap is not None:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test)
        if isinstance(shap_values, list):
            sv = np.asarray(shap_values[min(1, len(shap_values) - 1)], dtype=float)
        else:
            sv = np.asarray(shap_values, dtype=float)
            if sv.ndim == 3:
                sv = sv[:, :, min(1, sv.shape[2] - 1)]
        shap_top = np.max(np.abs(sv), axis=1) if sv.ndim == 2 else np.zeros(len(outs), dtype=float)
        shap_q90 = float(np.quantile(shap_top, 0.90)) if shap_top.size else 0.0
        shap_actions = [
            (
                _shap_guardrail_action(float(pp), float(uu), float(ii), int(rc), float(st), shap_q90)
                if use_equal_guardrail
                else _shap_guardrail_action_native(float(pp), float(st), shap_q90)
            )
            for pp, uu, ii, rc, st in zip(p, u, i_pre, chi_r_crit, shap_top)
        ]

    lime_actions: list[str] = []
    if LimeTabularExplainer is not None:
        pred_labels = np.array([int(o.get('raw_prediction', o.get('prediction', 0))) for o in outs], dtype=int)
        explain_idx = np.arange(min(len(x_test_np), MAX_EXPLAIN_ROWS))
        explain_mask = np.zeros(len(x_test_np), dtype=bool)
        explain_mask[explain_idx] = True
        lime_exp = LimeTabularExplainer(
            training_data=x_train_np,
            feature_names=list(x_model.columns),
            class_names=[str(v) for v in sorted(y.unique())],
            mode='classification',
            random_state=42,
            discretize_continuous=True,
        )
        lime_top_abs_arr = np.zeros(len(x_test_np), dtype=float)
        lime_mixed = np.zeros(len(x_test_np), dtype=bool)
        explained_top_abs = []
        for idx in explain_idx:
            row = x_test_np[idx]
            lbl = pred_labels[idx]
            try:
                e = lime_exp.explain_instance(row, model.predict_proba, num_features=min(8, x_train_np.shape[1]))
                weights = [float(w) for _, w in e.local_exp.get(int(lbl), [])]
            except Exception:
                weights = []
            top_abs = max([abs(w) for w in weights], default=0.0)
            has_pos = any(w > 0 for w in weights)
            has_neg = any(w < 0 for w in weights)
            lime_top_abs_arr[idx] = top_abs
            lime_mixed[idx] = bool(has_pos and has_neg)
            explained_top_abs.append(top_abs)
        lime_q90 = float(np.quantile(np.array(explained_top_abs, dtype=float), 0.90)) if explained_top_abs else 0.0
        lime_actions = [
            (
                _lime_guardrail_action(float(pp), float(uu), float(ii), int(rc), float(ta), lime_q90, bool(ms and mk))
                if use_equal_guardrail
                else _lime_guardrail_action_native(float(pp), float(ta), lime_q90, bool(ms and mk))
            )
            for pp, uu, ii, rc, ta, ms, mk in zip(p, u, i_pre, chi_r_crit, lime_top_abs_arr, lime_mixed, explain_mask)
        ]

    anchor_actions: list[str] = []
    if anchor_tabular is not None:
        explain_idx = np.arange(min(len(x_test_np), MAX_EXPLAIN_ROWS))
        explain_mask = np.zeros(len(x_test_np), dtype=bool)
        explain_mask[explain_idx] = True
        anchor_exp = anchor_tabular.AnchorTabularExplainer(
            class_names=[str(v) for v in sorted(y.unique())],
            feature_names=list(x_model.columns),
            train_data=x_train_np,
            categorical_names={},
            discretizer='quartile',
        )
        anchor_precision = np.ones(len(x_test_np), dtype=float)
        anchor_coverage = np.ones(len(x_test_np), dtype=float)
        anchor_empty = np.zeros(len(x_test_np), dtype=bool)
        for idx in explain_idx:
            row = x_test_np[idx]
            try:
                e = anchor_exp.explain_instance(
                    row,
                    model.predict,
                    threshold=0.95,
                    beam_size=1,
                    batch_size=100,
                )
                names = list(e.names())
                anchor_empty[idx] = len(names) == 0
                anchor_precision[idx] = float(e.precision())
                anchor_coverage[idx] = float(e.coverage())
            except Exception:
                anchor_empty[idx] = True
                anchor_precision[idx] = 0.0
                anchor_coverage[idx] = 0.0
        anchor_actions = [
            (
                _anchor_guardrail_action(float(pp), float(uu), float(ii), int(rc), float(pr), float(cv), bool(emp and mk))
                if use_equal_guardrail
                else _anchor_guardrail_action_native(float(pp), float(pr), float(cv), bool(emp and mk))
            )
            for pp, uu, ii, rc, pr, cv, emp, mk in zip(p, u, i_pre, chi_r_crit, anchor_precision, anchor_coverage, anchor_empty, explain_mask)
        ]

    rows = [
        {
            'baseline': 'full_observer_runtime',
            'information_access': 'full_structure',
            **_metrics(full_actions_runtime, expected_proxy, expected_reference, p, u, i_pre, chi_r, chi_r_crit),
        },
        {
            'baseline': 'full_observer_uncalibrated',
            'information_access': 'full_structure',
            **_metrics(full_actions_uncal, expected_proxy, expected_reference, p, u, i_pre, chi_r, chi_r_crit),
        },
        {
            'baseline': 'full_observer_calibrated',
            'information_access': 'full_structure',
            **_metrics(full_actions_cal, expected_proxy, expected_reference, p, u, i_pre, chi_r, chi_r_crit),
        },
        {
            'baseline': 'probability_threshold',
            'information_access': 'equal_guardrail' if use_equal_guardrail else 'native_risk_only',
            **_metrics(threshold_actions, expected_proxy, expected_reference, p, u, i_pre, chi_r, chi_r_crit),
        },
    ]
    if shap_actions:
        rows.append({
            'baseline': 'shap_guardrail',
            'information_access': 'equal_guardrail' if use_equal_guardrail else 'native_feature_importance',
            **_metrics(shap_actions, expected_proxy, expected_reference, p, u, i_pre, chi_r, chi_r_crit),
        })
    if lime_actions:
        rows.append({
            'baseline': 'lime_guardrail',
            'information_access': 'equal_guardrail' if use_equal_guardrail else 'native_local_surrogate',
            **_metrics(lime_actions, expected_proxy, expected_reference, p, u, i_pre, chi_r, chi_r_crit),
        })
    if anchor_actions:
        rows.append({
            'baseline': 'anchor_guardrail',
            'information_access': 'equal_guardrail' if use_equal_guardrail else 'native_rule_anchor',
            **_metrics(anchor_actions, expected_proxy, expected_reference, p, u, i_pre, chi_r, chi_r_crit),
        })
    notes = [
        'SHAP baseline is included via TreeExplainer.' if shap is not None else 'SHAP is not installed; SHAP baseline skipped.',
        'LIME baseline is included.' if LimeTabularExplainer is not None else 'LIME is not installed; LIME baseline skipped.',
        'Anchors baseline is included.' if anchor_tabular is not None else 'anchor-exp is not installed; Anchors baseline skipped.',
        'agreement_proxy is proxy-rule agreement, not clinical expert action accuracy.',
        'agreement_reference uses independent reference policy (risk + outcome context).',
    ]
    if dataset == 'breast_cancer':
        notes.append('For breast_cancer, reference is risk-dominated; probability-threshold baseline is expected to be strong.')

    out = {
        'dataset': dataset,
        'status': 'READY',
        'baseline_access': str(baseline_access),
        'rows': rows,
        'notes': notes,
    }
    out_dir = Path(out_root) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'baseline_comparison.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / f'baseline_comparison_{baseline_access}.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    md = [
        f'# Baseline comparison: {dataset}',
        '',
        f"- baseline_access: `{baseline_access}`",
        '',
        '| baseline | information_access | agreement_proxy | agreement_reference | missed_critical_ruptures | false_auto_accept_rate | false_block_rate | auto_accept_coverage |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        md.append(
            f"| {row['baseline']} | {row['information_access']} | {row['agreement_proxy']:.6f} | {row['agreement_reference']:.6f} | {row['missed_critical_ruptures']} | "
            f"{row['false_auto_accept_rate']:.6f} | {row['false_block_rate']:.6f} | {row['auto_accept_coverage']:.6f} |"
        )
    md += ['', *[f'- {n}' for n in out['notes']]]
    (out_dir / 'baseline_comparison.md').write_text('\n'.join(md), encoding='utf-8')
    (out_dir / f'baseline_comparison_{baseline_access}.md').write_text('\n'.join(md), encoding='utf-8')
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='breast_cancer')
    parser.add_argument('--out-root', default='reports/datasets')
    parser.add_argument('--baseline-access', default='native', choices=['native', 'equal_guardrail'])
    args = parser.parse_args()
    result = run(args.dataset, out_root=args.out_root, baseline_access=str(args.baseline_access))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
