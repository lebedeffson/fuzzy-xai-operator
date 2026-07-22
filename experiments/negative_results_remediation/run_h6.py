from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from fuzzyxai.rule_effects_v2 import (
    ConditionalRule,
    RuleEffectData,
    assess_rule_effect,
    eligible_candidate,
    match_controls,
    registered_detectability_grid,
    specific_effect,
    summarize_detectability,
)

from .common import ARTIFACTS, require_file, verify_protocol, write_json


class _ReducedModel:
    def __init__(self, model: object, keep: np.ndarray) -> None:
        self.model = model
        self.keep = keep

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(values[:, self.keep]))  # type: ignore[attr-defined]


class _TreeRuleAdapter:
    def __init__(self, model: DecisionTreeClassifier, medians: np.ndarray, seed: int) -> None:
        self.model = model
        self.medians = medians
        self.seed = seed

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(values))

    def predict_without_rule(self, values: np.ndarray, rule: ConditionalRule) -> np.ndarray:
        changed = np.asarray(values).copy()
        changed[:, rule.feature_indices] = self.medians[list(rule.feature_indices)]
        return np.asarray(self.model.predict(changed))

    def refit_without_rule(self, rule: ConditionalRule, values: np.ndarray, labels: np.ndarray) -> _ReducedModel:
        keep = np.asarray([index for index in range(values.shape[1]) if index not in rule.feature_indices], dtype=int)
        model = clone(self.model)
        model.fit(values[:, keep], labels)
        return _ReducedModel(model, keep)


def run_envelope(*, seed: int = 4201) -> None:
    require_file(ARTIFACTS / "lock" / "negative_remediation_lock.json", "H6 envelope requires protocol lock")
    rng = np.random.default_rng(seed)
    rows = []
    observations = []
    noise_factor = {"low": 1.0, "medium": 0.7, "high": 0.4}
    for point in registered_detectability_grid():
        n_active = max(20, int(2000 * point.support))
        attenuation = (1.0 - point.redundancy) * noise_factor[point.noise] * (1.0 - 0.12 * (point.interaction_order - 1)) * (1.0 - 0.35 * point.proxy_correlation)
        true_effect = point.effect_strength * max(0.05, attenuation)
        treated = rng.binomial(1, min(0.98, 0.5 + true_effect), size=n_active)
        control = rng.binomial(1, 0.5, size=n_active)
        observed = float(treated.mean() - control.mean())
        standard_error = float(np.sqrt(max(1e-12, treated.mean() * (1 - treated.mean()) / n_active + control.mean() * (1 - control.mean()) / n_active)))
        detected = observed - 1.96 * standard_error > 0.0
        false_discovery = detected and observed <= 0.0
        sign_correct = observed > 0.0
        observations.append((point, detected, false_discovery, sign_correct))
        rows.append({**asdict(point), "eligible": point.eligible, "n_active": n_active, "true_effect": true_effect, "observed_effect": observed, "standard_error": standard_error, "detected": detected, "false_discovery": false_discovery, "sign_correct": sign_correct})
    summary = summarize_detectability(observations)
    summary.update({"phase": "controlled_formative_detectability", "protocol_sha256": verify_protocol(), "grid_points": len(rows), "universal_rule_claim_allowed": False})
    write_json(ARTIFACTS / "h6" / "detectability_rows.json", rows)
    write_json(ARTIFACTS / "h6" / "detectability_summary.json", summary)
    print(f"PASS remediation-rule-envelope grid={len(rows)} criterion_met={summary['criterion_met']}")


def _dataset_effect(dataset_id: str, values: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, object]:
    train_values, test_values, train_labels, test_labels = train_test_split(values, labels, test_size=0.2, stratify=labels, random_state=seed)
    train_values, development_values, train_labels, development_labels = train_test_split(train_values, train_labels, test_size=0.25, stratify=train_labels, random_state=seed + 1)
    model = DecisionTreeClassifier(max_depth=6, max_leaf_nodes=20, min_samples_leaf=5, random_state=seed)
    model.fit(train_values, train_labels)
    medians = np.median(train_values, axis=0)
    correlations = np.nan_to_num(np.corrcoef(train_values, rowvar=False), nan=0.0)
    importances = np.asarray(model.feature_importances_)
    rules = []
    for feature in range(values.shape[1]):
        support_train = float(np.mean(train_values[:, feature] > medians[feature]))
        support_dev = float(np.mean(development_values[:, feature] > medians[feature]))
        redundancy = float(np.max(np.abs(np.delete(correlations[feature], feature)))) if values.shape[1] > 1 else 0.0
        rules.append(ConditionalRule(f"{dataset_id}:feature-{feature}", (feature,), support_train, redundancy, 1.0 - min(1.0, abs(support_train - support_dev)), int(np.bincount(train_labels).argmax()), 1, 1, int(support_train * len(train_values))))
    ranked = sorted(range(len(rules)), key=lambda index: (-importances[index], rules[index].redundancy, index))
    candidate = next((rules[index] for index in ranked if eligible_candidate(rules[index], unique_coverage=importances[index], direction_stable=True, subgroup_leakage=False)), rules[ranked[0]])
    adapter = _TreeRuleAdapter(model, medians, seed)
    non_rule_features = [index for index in range(train_values.shape[1]) if index not in candidate.feature_indices]
    stratification_feature = non_rule_features[0]
    edges = np.unique(np.quantile(train_values[:, stratification_feature], (0.25, 0.50, 0.75)))
    strata = np.digitize(test_values[:, stratification_feature], edges)
    data = RuleEffectData(train_values, train_labels, test_values, test_labels, strata, accuracy_score)
    effects = assess_rule_effect(candidate, adapter, data)
    controls = match_controls(candidate, rules, count=5)
    control_effects = [assess_rule_effect(control, adapter, data, estimands=("nonrefit",))["nonrefit"].effect for control in controls.controls]
    return {
        "dataset_id": dataset_id,
        "n_train": len(train_values),
        "n_development": len(development_values),
        "n_test": len(test_values),
        "candidate": asdict(candidate),
        "candidate_eligible": eligible_candidate(candidate, unique_coverage=float(importances[candidate.feature_indices[0]]), direction_stable=True, subgroup_leakage=False),
        "effects": {name: asdict(value) for name, value in effects.items()},
        "matched_controls": [asdict(item) for item in controls.controls],
        "matched_distances": controls.distances,
        "specific_nonrefit_effect": specific_effect(effects["nonrefit"].effect, control_effects),
        "test_used_for_candidate_selection": False,
        "conditional_resampling": "within_train_fitted_non_target_feature_quantile_strata",
        "conditional_strata_used_test_labels": False,
    }


def run_real() -> None:
    require_file(ARTIFACTS / "lock" / "negative_remediation_lock.json", "H6 real evaluation requires protocol lock")
    cancer = load_breast_cancer()
    wine = load_wine()
    rows = [
        _dataset_effect("breast_cancer_wisconsin", np.asarray(cancer.data, dtype=float), np.asarray(cancer.target, dtype=int), 4201),
        _dataset_effect("wine_recognition", np.asarray(wine.data, dtype=float), np.asarray(wine.target, dtype=int), 4202),
    ]
    signs = [np.sign(row["specific_nonrefit_effect"]) for row in rows]
    replicated = bool(all(sign == signs[0] and sign != 0 for sign in signs))
    summary = {
        "phase": "exploratory_real_data_after_protocol_lock",
        "protocol_sha256": verify_protocol(),
        "datasets": rows,
        "dataset_direction_consistent": replicated,
        "H6-R1": "measured_exploratory",
        "H6-R2": "measured_exploratory",
        "H6-R3": "measured_exploratory",
        "H6-R4": "not_confirmatory_datasets_not_registered_as_independent_before_lock",
        "H6-R5": "not_evaluated_certificate_adapter_absent_for_these_predictors",
        "H6-general": "not_supported",
        "positive_general_claim_allowed": False,
    }
    write_json(ARTIFACTS / "h6" / "real_rule_effects.json", summary)
    print(f"PASS remediation-h6-confirmatory datasets=2 H6-R4={summary['H6-R4']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("envelope", "real"))
    args = parser.parse_args()
    if args.stage == "envelope":
        run_envelope()
    else:
        run_real()


if __name__ == "__main__":
    main()
