from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .common import ARTIFACTS, canonical_bytes, git_commit, protocol, read_json, read_jsonl, sha256_bytes, sha256_file, verify_protocol_hash, write_json, write_jsonl


SIMPLE_POLICIES = (
    "max_confidence",
    "calibrated_confidence",
    "predictive_entropy",
    "weighted_linear",
    "explainer_disagreement",
    "simple_or",
    "provenance_only",
)
ALL_POLICIES = (
    "always_accept",
    *SIMPLE_POLICIES,
    "predictive_risk_P0",
    "full_fuzzyxai_P1",
    "full_fuzzyxai",
    "random_matched_budget",
)


def _load_joined(split: str) -> list[dict[str, Any]]:
    predictions = list(read_jsonl(ARTIFACTS / "predictions" / f"{split}.jsonl"))
    explanation_path = ARTIFACTS / "explanations" / f"{split}.jsonl"
    explanations = {str(row["object_id"]): row for row in read_jsonl(explanation_path)} if explanation_path.exists() else {}
    rows = []
    for prediction in predictions:
        explanation = explanations.get(str(prediction["object_id"]))
        row = dict(prediction)
        row.update(
            explanation_available=explanation is not None,
            ig_instability=None if explanation is None else 1.0 - float(explanation["ig_perturbation_stability"]),
            masking_instability=None if explanation is None else 1.0 - float(explanation["masking_perturbation_stability"]),
            explainer_disagreement=None if explanation is None else 1.0 - float(explanation["explainer_top_k_agreement"]),
        )
        rows.append(row)
    return rows


def _base_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                1.0 - float(row["confidence"]),
                float(row["entropy"]),
                1.0 - float(row["margin"]),
            ]
            for row in rows
        ],
        dtype=float,
    )


def _route_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values = []
    for row in rows:
        missing = not bool(row["explanation_available"])
        values.append(
            [
                1.0 if missing else float(row["ig_instability"]),
                1.0 if missing else float(row["masking_instability"]),
                1.0 if missing else float(row["explainer_disagreement"]),
                float(missing),
            ]
        )
    return np.asarray(values, dtype=float)


def _fit_scores(validation: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    labels = np.asarray([int(row["label"]) for row in validation])
    predictions = np.asarray([int(row["prediction"]) for row in validation])
    invalid = (labels != predictions).astype(int)
    confidence = np.asarray([float(row["confidence"]) for row in validation])
    test_confidence = np.asarray([float(row["confidence"]) for row in test])
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(confidence, 1 - invalid)

    base_validation = _base_matrix(validation)
    base_test = _base_matrix(test)
    route_validation = _route_matrix(validation)
    route_test = _route_matrix(test)
    p0 = LogisticRegression(max_iter=1000, random_state=1301, class_weight="balanced").fit(base_validation, invalid)
    p1 = LogisticRegression(max_iter=1000, random_state=1301, class_weight="balanced").fit(np.column_stack((base_validation, route_validation)), invalid)
    calibrated_validation = 1.0 - calibrator.predict(confidence)
    calibrated_test = 1.0 - calibrator.predict(test_confidence)

    def build(rows: Sequence[Mapping[str, Any]], base: np.ndarray, route: np.ndarray, calibrated: np.ndarray, p0_scores: np.ndarray, p1_scores: np.ndarray) -> dict[str, np.ndarray]:
        confidence_risk = base[:, 0]
        entropy = base[:, 1]
        margin_risk = base[:, 2]
        disagreement = route[:, 2]
        provenance = route[:, 3]
        weighted = 0.50 * confidence_risk + 0.30 * entropy + 0.20 * margin_risk
        thresholds = {
            "confidence": float(np.quantile(base_validation[:, 0], 0.80)),
            "entropy": float(np.quantile(base_validation[:, 1], 0.80)),
            "disagreement": float(np.quantile(route_validation[:, 2], 0.80)),
            "provenance": 0.5,
        }
        simple_or = (
            (confidence_risk >= thresholds["confidence"]).astype(float)
            + (entropy >= thresholds["entropy"]).astype(float)
            + (disagreement >= thresholds["disagreement"]).astype(float)
            + (provenance >= thresholds["provenance"]).astype(float)
        ) / 4.0
        full = 1.0 - (1.0 - p1_scores) * (1.0 - np.maximum.reduce((route[:, 0], route[:, 1], route[:, 2], route[:, 3])))
        random_scores = np.random.default_rng(1301).random(len(rows))
        return {
            "always_accept": np.zeros(len(rows)),
            "max_confidence": confidence_risk,
            "calibrated_confidence": calibrated,
            "predictive_entropy": entropy,
            "weighted_linear": weighted,
            "explainer_disagreement": disagreement,
            "simple_or": simple_or,
            "provenance_only": provenance,
            "predictive_risk_P0": p0_scores,
            "full_fuzzyxai_P1": p1_scores,
            "full_fuzzyxai": full,
            "random_matched_budget": random_scores,
        }

    validation_scores = build(
        validation,
        base_validation,
        route_validation,
        calibrated_validation,
        p0.predict_proba(base_validation)[:, 1],
        p1.predict_proba(np.column_stack((base_validation, route_validation)))[:, 1],
    )
    test_scores = build(
        test,
        base_test,
        route_test,
        calibrated_test,
        p0.predict_proba(base_test)[:, 1],
        p1.predict_proba(np.column_stack((base_test, route_test)))[:, 1],
    )
    fit = {
        "isotonic_x": [float(value) for value in calibrator.X_thresholds_],
        "isotonic_y": [float(value) for value in calibrator.y_thresholds_],
        "p0_coefficients": p0.coef_.tolist(),
        "p0_intercept": p0.intercept_.tolist(),
        "p1_coefficients": p1.coef_.tolist(),
        "p1_intercept": p1.intercept_.tolist(),
        "selected_without_test_labels": True,
    }
    return validation_scores, test_scores, fit


def _actions(scores: np.ndarray, budget: float, policy_name: str | None = None) -> np.ndarray:
    actions = np.full(len(scores), "accept", dtype=object)
    if policy_name == "always_accept":
        return actions
    count = int(round(budget * len(scores)))
    order = np.argsort(-scores, kind="stable")
    actions[order[:count]] = "review"
    return actions


def _expected_calibration_error(invalid: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    clipped = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    targets = np.asarray(invalid, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        mask = (clipped >= edges[index]) & (
            (clipped <= edges[index + 1]) if upper_inclusive else (clipped < edges[index + 1])
        )
        if mask.any():
            error += float(mask.mean()) * abs(float(targets[mask].mean()) - float(clipped[mask].mean()))
    return error


def _metrics(
    rows: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    invalid: np.ndarray,
    budget: float,
    costs: Mapping[str, float],
    policy_name: str | None = None,
) -> dict[str, object]:
    actions = _actions(scores, budget, policy_name)
    accepted = actions == "accept"
    reviewed = actions == "review"
    wrong = accepted & invalid
    risk = float(wrong.sum() / max(1, accepted.sum()))
    return {
        "automatic_coverage": float(accepted.mean()),
        "accepted_error_rate": risk,
        "wrong_automatic_actions": int(wrong.sum()),
        "selective_risk": risk,
        "manual_review_load": int(reviewed.sum()),
        "review_rate": float(reviewed.mean()),
        "false_blocks": 0,
        "true_structural_blocks": 0,
        "total_cost": float(wrong.sum() * float(costs["wrong_accept"]) + reviewed.sum() * float(costs["review"])),
        "risk_auroc": float(roc_auc_score(invalid, scores)),
        "risk_auprc": float(average_precision_score(invalid, scores)),
        "expected_calibration_error": _expected_calibration_error(invalid, scores),
        "brier_score": float(brier_score_loss(invalid, np.clip(scores, 0.0, 1.0))),
        "action_sha256": sha256_bytes(canonical_bytes(actions.tolist())),
    }


def pre_score() -> dict[str, object]:
    cfg = protocol()
    validation = _load_joined("validation")
    test = _load_joined("sealed_test")
    validation_scores, test_scores, fit = _fit_scores(validation, test)
    invalid_validation = np.asarray([not bool(row["is_correct"]) for row in validation])
    budget = float(cfg["policy_evaluation"]["primary_budget"])
    validation_metrics = {
        name: _metrics(
            validation,
            scores,
            invalid_validation,
            budget,
            cfg["policy_evaluation"]["cost_profiles"]["balanced"],
            name,
        )
        for name, scores in validation_scores.items()
    }
    best_simple = min(SIMPLE_POLICIES, key=lambda name: (validation_metrics[name]["wrong_automatic_actions"], name))

    rows = []
    for index, item in enumerate(test):
        rows.append(
            {
                "object_id": item["object_id"],
                "prediction": item["prediction"],
                "confidence": item["confidence"],
                "scores": {name: float(values[index]) for name, values in test_scores.items()},
                "explanation_available": item["explanation_available"],
            }
        )
    score_path = ARTIFACTS / "policies" / "test_policy_scores.jsonl"
    write_jsonl(score_path, rows)
    write_json(ARTIFACTS / "policies" / "validation_selection.json", {"best_simple_policy": best_simple, "primary_budget": budget, "metrics": validation_metrics})
    write_json(ARTIFACTS / "policies" / "fitted_models.json", fit)
    lock = {
        "stage": "pre_score_complete",
        "test_labels_loaded": False,
        "policy_scores_sha256": sha256_file(score_path),
        "validation_selection_sha256": sha256_file(ARTIFACTS / "policies" / "validation_selection.json"),
        "best_simple_policy": best_simple,
        "protocol_sha256": verify_protocol_hash(),
    }
    write_json(ARTIFACTS / "policies" / "pre_score_lock.json", lock)
    return lock


def _holm(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.zeros(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def _bootstrap(full_wrong: np.ndarray, baseline_wrong: np.ndarray, repetitions: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(full_wrong)
    observed = float((baseline_wrong.astype(float) - full_wrong.astype(float)).mean())
    differences = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, 250):
        size = min(250, repetitions - start)
        indices = rng.integers(0, n, size=(size, n))
        differences[start : start + size] = (baseline_wrong[indices].mean(axis=1) - full_wrong[indices].mean(axis=1))
    lower, upper = np.quantile(differences, [0.025, 0.975])
    lower_tail = int(np.count_nonzero(differences <= 0.0))
    upper_tail = int(np.count_nonzero(differences >= 0.0))
    p_value = min(1.0, 2.0 * (min(lower_tail, upper_tail) + 1) / (repetitions + 1))
    return {"absolute_rate_reduction": observed, "ci_lower": float(lower), "ci_upper": float(upper), "p_value": p_value}


def score(*, recovery: bool = False) -> dict[str, object]:
    cfg = protocol()
    opening_path = ARTIFACTS / "policies" / "scoring_opening_record.json"
    completion_path = ARTIFACTS / "policies" / "scoring_completion.json"
    summary_path = ARTIFACTS / "policies" / "summary.json"
    recovery_completion_path = ARTIFACTS / "policies" / "scoring_recovery_completion.json"
    recovery_lock_path = ARTIFACTS / "policies" / "scoring_recovery_lock.json"
    if recovery_completion_path.exists():
        recovery_completion = read_json(recovery_completion_path)
        if recovery_completion["summary_sha256"] != sha256_file(summary_path):
            raise RuntimeError("recovered scoring summary no longer matches its lock")
        return read_json(summary_path)
    if completion_path.exists() and not recovery:
        completion = read_json(completion_path)
        if completion["summary_sha256"] != sha256_file(summary_path):
            raise RuntimeError("completed scoring summary no longer matches its lock")
        return read_json(summary_path)
    if opening_path.exists() and not recovery:
        raise RuntimeError("scoring was already opened but did not complete; rerun is forbidden")
    lock_path = ARTIFACTS / "policies" / "pre_score_lock.json"
    lock = read_json(lock_path)
    score_path = ARTIFACTS / "policies" / "test_policy_scores.jsonl"
    if lock["policy_scores_sha256"] != sha256_file(score_path):
        raise RuntimeError("policy scores changed after pre-score lock")
    label_vault_path = ARTIFACTS / "private" / "sealed_test_labels.json"
    if recovery:
        invalid_dir = ARTIFACTS / "policies" / "invalid_scoring_run_1"
        invalid_marker = invalid_dir / "invalid_marker.json"
        if not invalid_marker.exists() or not (invalid_dir / "scoring_completion.json").exists():
            raise RuntimeError("scoring recovery requires the preserved invalid run")
        write_json(
            recovery_lock_path,
            {
                "stage": "scoring_only_recovery_about_to_open_existing_vault",
                "invalid_marker_sha256": sha256_file(invalid_marker),
                "original_completion_sha256": sha256_file(invalid_dir / "scoring_completion.json"),
                "pre_score_lock_sha256": sha256_file(lock_path),
                "policy_scores_sha256": sha256_file(score_path),
                "label_vault_sha256": sha256_file(label_vault_path),
                "protocol_sha256": verify_protocol_hash(),
                "recovery_code_commit": git_commit(),
                "allowed_changes": ["always_accept_action_semantics", "finite_bootstrap_add_one_p_value"],
                "policy_scores_thresholds_and_selected_baseline_changed": False,
            },
        )
    else:
        write_json(
            opening_path,
            {
                "stage": "test_labels_about_to_open_for_scoring_only",
                "pre_score_lock_sha256": sha256_file(lock_path),
                "policy_scores_sha256": sha256_file(score_path),
                "label_vault_sha256": sha256_file(label_vault_path),
                "protocol_sha256": verify_protocol_hash(),
                "scoring_code_commit": git_commit(),
            },
        )
    rows = list(read_jsonl(score_path))
    label_vault = read_json(label_vault_path)["labels"]
    labels = np.asarray([int(label_vault[str(row["object_id"])]) for row in rows])
    predictions = np.asarray([int(row["prediction"]) for row in rows])
    invalid = labels != predictions
    budgets = [float(value) for value in cfg["policy_evaluation"]["review_budgets"]]
    raw_rows: list[dict[str, object]] = []
    action_indicators: dict[tuple[str, float], np.ndarray] = {}
    for policy_name in ALL_POLICIES:
        scores = np.asarray([float(row["scores"][policy_name]) for row in rows])
        for budget in budgets:
            actions = _actions(scores, budget, policy_name)
            wrong = (actions == "accept") & invalid
            action_indicators[(policy_name, budget)] = wrong
            for cost_name, costs in cfg["policy_evaluation"]["cost_profiles"].items():
                raw_rows.append(
                    {
                        "policy": policy_name,
                        "review_budget": budget,
                        "cost_profile": cost_name,
                        **_metrics(rows, scores, invalid, budget, costs, policy_name),
                    }
                )
    raw_path = ARTIFACTS / "policies" / "policy_results.csv"
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)

    comparisons = []
    best_simple = str(lock["best_simple_policy"])
    for budget in budgets:
        full = action_indicators[("full_fuzzyxai", budget)]
        for baseline in SIMPLE_POLICIES + ("predictive_risk_P0", "random_matched_budget"):
            result = _bootstrap(full, action_indicators[(baseline, budget)], int(cfg["statistics"]["paired_bootstrap_repetitions"]), 1301 + int(100 * budget))
            baseline_wrong = int(action_indicators[(baseline, budget)].sum())
            full_wrong = int(full.sum())
            result.update(
                review_budget=budget,
                baseline=baseline,
                full_wrong=full_wrong,
                baseline_wrong=baseline_wrong,
                relative_reduction=float((baseline_wrong - full_wrong) / max(1, baseline_wrong)),
                primary=budget == cfg["policy_evaluation"]["primary_budget"] and baseline == best_simple,
            )
            comparisons.append(result)
    adjusted = _holm([float(row["p_value"]) for row in comparisons])
    for row, value in zip(comparisons, adjusted, strict=True):
        row["holm_adjusted_p"] = value
    write_json(ARTIFACTS / "policies" / "statistical_tests.json", {"comparisons": comparisons, "bootstrap_repetitions": cfg["statistics"]["paired_bootstrap_repetitions"]})

    test_quality = {
        "objects": len(rows),
        "accuracy": float((~invalid).mean()),
        "errors": int(invalid.sum()),
        "labels_opened_only_for_scoring": True,
        "pre_score_lock_sha256": sha256_file(lock_path),
        "best_simple_selected_on_validation": best_simple,
    }
    write_json(ARTIFACTS / "policies" / "test_quality.json", test_quality)
    primary = next(row for row in comparisons if row["primary"])
    summary = {"test_quality": test_quality, "primary_comparison": primary, "policy_results_sha256": sha256_file(raw_path)}
    write_json(summary_path, summary)
    completion = {
        "stage": "scoring_recovery_complete" if recovery else "scoring_complete",
        "opening_record_sha256": sha256_file(recovery_lock_path if recovery else opening_path),
        "summary_sha256": sha256_file(summary_path),
        "policy_results_sha256": sha256_file(raw_path),
        "statistical_tests_sha256": sha256_file(ARTIFACTS / "policies" / "statistical_tests.json"),
        "post_open_tuning": False,
        "scoring_only_recovery": recovery,
    }
    write_json(recovery_completion_path if recovery else completion_path, completion)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pre-score", "score", "score-recovery"), required=True)
    args = parser.parse_args()
    result = pre_score() if args.stage == "pre-score" else score(recovery=args.stage == "score-recovery")
    print(f"PASS: policy stage={args.stage} {result}")


if __name__ == "__main__":
    main()
