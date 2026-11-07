"""Pre-registered split / controlled data-corruption study on WDBC.

This evaluates a selective-application policy around a predictive model. It
uses the *canonical* FuzzyXAI risk observer but does not claim to test the full
semantic explanation, reduction or certification stack. Features called
"technical_completeness_proxy" are not I_pre from the dissertation.

The labels of the real WDBC data are used for evaluation only. Corruption
labels are hidden from the policies; missing-value and range checks operate on
observable inputs and training-set statistics only. No clinical safety claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / 'framework' / 'fuzzyxai'
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

WEIGHTS = {'rho_p': 0.35, 'u_M': 0.20, 'one_minus_I_pre': 0.20, 'Delta': 0.20, 'chi_R': 0.05}
TARGET_ACCEPT_COVERAGE = 0.70
STRESSORS = ('clean', 'missing_10pct', 'missing_40pct', 'sensor_noise_2sd')
SEED_DEFAULT = (42, 123, 2026)


@dataclass(frozen=True)
class Prepared:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    ids_train: np.ndarray
    ids_val: np.ndarray
    ids_test: np.ndarray
    medians: np.ndarray
    stdev: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    dataset_sha256: str


def _sha(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def prepare(seed: int) -> Prepared:
    """Split before estimating medians, scales and range thresholds."""
    data = load_breast_cancer()
    x = np.asarray(data.data, dtype=np.float64)
    y = np.asarray(data.target, dtype=np.int64)
    ids = np.arange(len(y))
    x_tv, x_test, y_tv, y_test, ids_tv, ids_test = train_test_split(
        x, y, ids, test_size=0.20, stratify=y, random_state=seed)
    x_train, x_val, y_train, y_val, ids_train, ids_val = train_test_split(
        x_tv, y_tv, ids_tv, test_size=0.25, stratify=y_tv, random_state=seed + 1)
    medians = np.median(x_train, axis=0)
    stdev = np.maximum(np.std(x_train, axis=0), 1e-9)
    lower, upper = np.quantile(x_train, [0.005, 0.995], axis=0)
    assert not set(ids_test) & set(ids_train) and not set(ids_test) & set(ids_val)
    assert not set(ids_val) & set(ids_train)
    return Prepared(
        x_train, x_val, x_test, y_train, y_val, y_test,
        ids_train, ids_val, ids_test, medians, stdev, lower, upper,
        _sha(x.tobytes() + y.tobytes()),
    )


def corrupt(x: np.ndarray, kind: str, rng: np.random.Generator, data: Prepared) -> tuple[np.ndarray, np.ndarray]:
    """Controlled alteration of *inputs*, preserving ground-truth labels."""
    output = np.asarray(x, dtype=float).copy()
    missing = np.zeros_like(output, dtype=bool)
    if kind == 'clean':
        pass
    elif kind in {'missing_10pct', 'missing_40pct'}:
        n_drop = 3 if kind == 'missing_10pct' else 12
        for i in range(len(output)):
            indices = rng.choice(output.shape[1], size=n_drop, replace=False)
            missing[i, indices] = True
        output[missing] = np.nan
    elif kind == 'sensor_noise_2sd':
        # This is a simulated artifact, not actual sensor hardware noise.
        output = output + rng.normal(0.0, 2.0, size=output.shape) * data.stdev
    else:
        raise ValueError(f'unknown perturbation: {kind}')
    return output, missing


def _normalize_entropy(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1) / math.log(p.shape[1])


def features_and_proba(model: Any, x_observed: np.ndarray, missing: np.ndarray, data: Prepared) -> dict[str, np.ndarray]:
    if x_observed.ndim != 2 or x_observed.shape[1] != len(data.medians):
        raise ValueError('wrong number of features')
    if missing.shape != x_observed.shape or np.any(np.isinf(x_observed)):
        raise ValueError('invalid observed feature data')
    if np.any(np.isnan(x_observed) != missing):
        raise ValueError('missing mask disagrees with NaN observations')
    filled = np.where(missing, data.medians[None, :], x_observed)
    proba = model.predict_proba(filled)
    score = np.max(proba, axis=1)
    missing_fraction = np.mean(missing, axis=1)
    # Out-of-training-range fraction is an OBSERVABLE diagnostic proxy.
    outliers = (~missing) & ((x_observed < data.lower) | (x_observed > data.upper))
    outlier_fraction = np.mean(outliers, axis=1)
    confidence_risk = 1.0 - score
    entropy = _normalize_entropy(proba)
    critical = (missing_fraction >= 0.30) | (outlier_fraction >= 0.45)
    return {
        'proba': proba, 'prediction': np.argmax(proba, axis=1),
        'confidence': score, 'rho_p': confidence_risk, 'u_M': entropy,
        'missing_fraction': missing_fraction, 'outlier_fraction': outlier_fraction,
        'critical': critical,
    }


def _risk_components(obs: dict[str, np.ndarray], i: int) -> dict[str, float]:
    # These are deliberately proxies, not a claim about semantic I_pre or Delta.
    missing = float(obs['missing_fraction'][i])
    outlier = float(obs['outlier_fraction'][i])
    return {
        'rho_p': float(obs['rho_p'][i]),
        'u_M': float(obs['u_M'][i]),
        'one_minus_I_pre': max(missing, outlier),  # technical completeness proxy
        'Delta': missing,                         # missing-input information proxy
        'chi_R': float(obs['critical'][i]),
    }


def _raw_risks(obs: dict[str, np.ndarray]) -> np.ndarray:
    from fuzzyxai.core.risk_observer import compute_canonical_risk
    return np.array([compute_canonical_risk(_risk_components(obs, i), WEIGHTS) for i in range(len(obs['confidence']))])


def _acceptance_threshold(scores: np.ndarray, target_coverage: float, *, lower_better: bool) -> float:
    if not 0.0 < target_coverage < 1.0:
        raise ValueError('target_coverage must be in (0,1)')
    ranked = np.sort(np.asarray(scores, dtype=float))
    k = max(1, min(len(ranked)-1, int(np.ceil(len(ranked) * target_coverage))))
    if lower_better:
        # Strictly-below boundary, inclusive of the kth candidate when possible.
        kth = ranked[k-1]
        next_val = ranked[k] if k < len(ranked) else np.inf
        return float(np.nextafter(kth, np.inf) if next_val == kth else (kth + next_val)/2)
    reverse = ranked[::-1]
    kth = reverse[k-1]
    next_val = reverse[k] if k < len(reverse) else -np.inf
    return float(np.nextafter(kth, -np.inf) if next_val == kth else (kth + next_val)/2)


def calibrated_thresholds(val_obs: dict[str, np.ndarray]) -> dict[str, float]:
    valid = ~val_obs['critical']
    risks = _raw_risks(val_obs)[valid]
    low = _acceptance_threshold(risks, TARGET_ACCEPT_COVERAGE, lower_better=True)
    low = min(low, 0.77)
    # Only theta_1 is used for the accept/not-accept comparison below.
    # Keep action thresholds strictly increasing (not silently normalized).
    remaining = (1.0 - low) / 5.0
    return {
        'theta_1': low,
        'theta_2': low + remaining,
        'theta_3': low + 2 * remaining,
        'theta_4': low + 3 * remaining,
    }


def decisions(obs: dict[str, np.ndarray], confidence_cut: float, thresholds: dict[str, float]) -> dict[str, np.ndarray]:
    from fuzzyxai.core.risk_observer import observe_risk

    confidence_accept = obs['confidence'] >= confidence_cut
    equal_guardrail_accept = confidence_accept & ~obs['critical']
    actions: list[str] = []
    risks: list[float] = []
    for i in range(len(confidence_accept)):
        components = _risk_components(obs, i)
        result = observe_risk(components, WEIGHTS, thresholds, int(obs['critical'][i]))
        actions.append(result.action)
        risks.append(float(result.rho))
        if obs['critical'][i] and result.action != 'block':
            raise AssertionError('Critical diagnostic did not override the risk action')
    return {
        'baseline_accept': confidence_accept,
        'equal_guardrail_accept': equal_guardrail_accept,
        'observer_accept': np.array(actions) == 'accept',
        'actions': np.asarray(actions),
        'rho': np.asarray(risks),
    }


def group_metrics(
    obs: dict[str, np.ndarray],
    dec: dict[str, np.ndarray],
    y: np.ndarray,
    kind: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for name, accepts in [
        ('confidence_only', dec['baseline_accept']),
        ('confidence_equal_guardrail', dec['equal_guardrail_accept']),
        ('canonical_risk_with_qc', dec['observer_accept']),
    ]:
        n_accept = int(np.sum(accepts))
        pred = obs['prediction']
        n_bad_accept = int(np.sum(accepts & obs['critical']))
        rows.append({
            'seed': seed, 'perturbation': kind, 'policy': name,
            'n': int(len(y)), 'n_auto_accept': n_accept,
            'coverage': n_accept / len(y),
            'accepted_accuracy': float(np.mean(pred[accepts] == y[accepts])) if n_accept else None,
            'accepted_error_rate': float(np.mean(pred[accepts] != y[accepts])) if n_accept else None,
            'critical_flagged_by_observable_qc': int(np.sum(obs['critical'])),
            'auto_accepted_with_critical_flag': n_bad_accept,
            'incorrect_accepted_predictions': int(np.sum(accepts & (pred != y))),
        })
    return rows


def paired_bootstrap_difference(
    a: np.ndarray, b: np.ndarray, seed: int, n_boot: int = 1500,
) -> dict[str, Any]:
    """Difference in binary bad-accept flags; rows paired by held-out object."""
    if len(a) != len(b) or len(a) == 0:
        raise ValueError('bootstrap observations must be paired and nonempty')
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(diffs[rng.integers(0, len(diffs), size=len(diffs))]) for _ in range(n_boot)])
    return {
        'paired_mean_difference': float(diffs.mean()),
        'ci95_low': float(np.quantile(boot, 0.025)),
        'ci95_high': float(np.quantile(boot, 0.975)),
        'bootstrap_repetitions': n_boot,
        'unit': 'held_out_row',
    }


def one_split(seed: int, n_boot: int = 1500) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    p = prepare(seed)
    model = RandomForestClassifier(n_estimators=120, max_depth=None, random_state=seed, n_jobs=1)
    model.fit(p.x_train, p.y_train)
    val_observed = np.asarray(p.x_val)
    val_obs = features_and_proba(model, val_observed, np.zeros_like(val_observed, dtype=bool), p)
    confidence_cut = _acceptance_threshold(val_obs['confidence'], TARGET_ACCEPT_COVERAGE, lower_better=False)
    theta = calibrated_thresholds(val_obs)
    if len(set(map(float, theta.values()))) != 4:
        raise AssertionError('observer thresholds not strictly increasing')

    results: list[dict[str, Any]] = []
    per_case: list[dict[str, Any]] = []
    # Clean data remains entirely held out for model-quality measurement.
    clean = features_and_proba(model, p.x_test, np.zeros_like(p.x_test, dtype=bool), p)
    clean_auc = float(roc_auc_score(p.y_test, clean['proba'][:, 1]))
    clean_acc = float(accuracy_score(p.y_test, clean['prediction']))
    clean_f1 = float(f1_score(p.y_test, clean['prediction']))

    for perturbation in STRESSORS:
        # Reinitialize RNG for every condition (independent but deterministic).
        condition_seed = seed * 100 + STRESSORS.index(perturbation) + 17
        x_observed, missing = corrupt(p.x_test, perturbation, np.random.default_rng(condition_seed), p)
        obs = features_and_proba(model, x_observed, missing, p)
        dec = decisions(obs, confidence_cut, theta)
        results.extend(group_metrics(obs, dec, p.y_test, perturbation, seed))
        for i in range(len(p.y_test)):
            per_case.append({
                'seed': seed, 'case_hash': _sha(f'{p.dataset_sha256}:{seed}:{int(p.ids_test[i])}'.encode())[:16],
                'condition': perturbation, 'y': int(p.y_test[i]),
                'pred': int(obs['prediction'][i]),
                'max_probability': round(float(obs['confidence'][i]), 6),
                'missing_fraction': round(float(obs['missing_fraction'][i]), 6),
                'outlier_fraction': round(float(obs['outlier_fraction'][i]), 6),
                'chi_R_crit': int(obs['critical'][i]),
                'canonical_rho': round(float(dec['rho'][i]), 6),
                'confidence_only_auto_accept': int(dec['baseline_accept'][i]),
                'equal_guardrail_auto_accept': int(dec['equal_guardrail_accept'][i]),
                'risk_qc_auto_accept': int(dec['observer_accept'][i]),
                'risk_qc_action': str(dec['actions'][i]),
            })
    poor = [r for r in per_case if r['condition'] == 'missing_40pct']
    bootstrap = paired_bootstrap_difference(
        np.array([r['confidence_only_auto_accept'] * r['chi_R_crit'] for r in poor]),
        np.array([r['risk_qc_auto_accept'] * r['chi_R_crit'] for r in poor]), seed + 10001, n_boot=n_boot,
    )
    shared_guardrail_diff = paired_bootstrap_difference(
        np.array([r['equal_guardrail_auto_accept'] for r in poor]),
        np.array([r['risk_qc_auto_accept'] for r in poor]), seed + 10003, n_boot=n_boot,
    )
    split = {
        'seed': seed, 'dataset': 'sklearn Breast Cancer Wisconsin Diagnostic',
        'dataset_sha256': p.dataset_sha256,
        'n_training': len(p.y_train), 'n_validation': len(p.y_val), 'n_holdout': len(p.y_test),
        'disjoint_splits': True,
        'model': 'RandomForestClassifier', 'model_n_estimators': 120,
        'model_accuracy_clean': clean_acc, 'model_roc_auc_clean': clean_auc,
        'model_f1_clean': clean_f1,
        'confidence_only_cutoff_from_validation': confidence_cut,
        'canonical_risk_thresholds_from_validation': theta,
        'validation_y_used_to_calibrate_policy': False,
        'target_clean_acceptance_coverage': TARGET_ACCEPT_COVERAGE,
        'paired_bootstrap_critical_flag_false_accept_difference_missing_40pct': bootstrap,
        'equal_guardrail_vs_observer_accept_difference_missing_40pct': shared_guardrail_diff,
    }
    return split, results, per_case


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def run(out_dir: Path, seeds: tuple[int, ...] = SEED_DEFAULT, n_boot: int = 1500) -> dict[str, Any]:
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError('seeds must be unique nonnegative integers')
    if n_boot < 100:
        raise ValueError('n_boot must be >=100')
    split_results, metrics, cases = [], [], []
    for seed in seeds:
        split, current_metrics, current_cases = one_split(seed, n_boot=n_boot)
        split_results.append(split)
        metrics.extend(current_metrics)
        cases.extend(current_cases)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / 'group_metrics.csv', metrics)
    _write_csv(out_dir / 'heldout_cases.csv', cases)
    summary = {
        'protocol': 'chapter5_wdbc_selective_application_stress_v1',
        'seeds': list(seeds), 'sklearn_version': SKLEARN_VERSION,
        'source_role': 'public_real_tabular_features_with_controlled_test_perturbation',
        'predictive_task': 'breast_cancer_binary_label',
        'not_a_clinical_deployment_or_ecg_study': True,
        'uses_canonical_fuzzyxai_risk_observer': True,
        'full_semantic_explanation_route_validated': False,
        'technical_completeness_is_not_semantic_I_pre': True,
        'delta_is_missingness_proxy_not_proved_reduction_loss': True,
        'critical_reference_derived_from_observed_mask_or_range_threshold': True,
        'confidence_only_baseline_has_no_quality_access': True,
        'equal_guardrail_baseline_shares_the_same_critical_flag': True,
        'clean_coverage_calibrated_on_unlabelled_validation_features': True,
        'standalone_confidence_comparison_not_causal_operator_advantage': True,
        'equal_guardrail_comparison_is_information_matched_for_critical_flag_only': True,
        'controlled_corruption_labels_are_not_passed_to_policy': True,
        'clinical_metrics': None,
        'n_test_cases_total': len(cases), 'n_split_seeds': len(seeds),
        'splits': split_results,
        'claim_scope': 'Component-level P19 risk observer demonstration; synthetic faults imposed on real held-out WDBC rows.',
        'limitations': [
            'Confidence-only has no input-quality flags; compare to equal-guardrail for a shared flag. This cannot identify the operator contribution alone.',
            'No clinical decision outcomes or independent adjudication of diagnostic quality.',
            'Missingness and outlier fractions are only engineering proxies for semantic explanation quantities.',
            'Synthetic corruptions and calibration thresholds may not generalize to unseen natural measurement faults.',
            'The public WDBC dataset does not supply patient-group IDs for patient-wise split verification.',
        ],
    }
    _write_json(out_dir / 'summary.json', summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'reports/chapter5/wdbc_selective_application')
    ap.add_argument('--seeds', type=str, default=','.join(map(str, SEED_DEFAULT)))
    ap.add_argument('--bootstrap', type=int, default=1500)
    args = ap.parse_args()
    seeds = tuple(int(x.strip()) for x in args.seeds.split(',') if x.strip())
    summary = run(args.out, seeds=seeds, n_boot=args.bootstrap)
    print(json.dumps({'protocol':summary['protocol'], 'n_cases':summary['n_test_cases_total'], 'splits':len(summary['splits']), 'out':str(args.out)}, indent=2))


if __name__ == '__main__':
    main()
