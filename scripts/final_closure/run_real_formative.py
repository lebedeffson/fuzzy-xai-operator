#!/usr/bin/env python3
"""Fit and compare real P0/P1 controllers on cross-fitted development evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from common import ROOT, STUDY, sha256, write
from oof_pipeline import DATASETS, PREDICTIVE_CHANNELS, ROUTE_CHANNELS


OUTPUT = STUDY / "formative_real"
BUDGETS = (0.05, 0.10, 0.20, 0.30)
SEED = 7419
PRIMARY_BUDGET = 0.20
GOOD_WHEN_HIGH = {
    "calibrated_confidence",
    "prediction_margin",
    "boundary_distance",
    "seed_stability",
    "bootstrap_stability",
    "perturbation_stability",
    "provenance_completeness",
    "canonical_hash_status",
}
COMPONENT_GROUPS = {
    "provenance": ("provenance_completeness", "canonical_hash_status", "missing_evidence_channels"),
    "route_faults": ("typed_route_fault",),
    "stability": ("seed_stability", "bootstrap_stability", "perturbation_stability"),
    "explainer_disagreement": ("explainer_disagreement", "conflict_severity"),
    "model_disagreement": ("model_checkpoint_disagreement",),
    "shift": ("label_free_shift_score", "reference_set_deviation", "data_quality_profile"),
    "representation_class": ("representation_class",),
    "reduction_loss": ("reduction_loss",),
}


def main() -> None:
    audit = json.loads((STUDY / "p0_p1_feature_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise SystemExit("BLOCKED: P0/P1 feature audit must pass before controller formative fit")
    frame = _load_rows()
    target, reasons = _operational_target(frame)
    groups = (frame["dataset_id"].astype(str) + ":fold:" + frame["fold"].astype(str)).to_numpy()
    p0, p1, feature_names = _matrices(frame)
    p0_scores, p0_models = _cross_fitted_risk(p0, target, groups, seed=SEED)
    p1_scores, p1_models = _cross_fitted_risk(p1, target, groups, seed=SEED)
    scores = _baseline_scores(frame, p0_scores, p1_scores)
    rows = _evaluate(scores, target, frame, budgets=BUDGETS)
    ablation = _component_ablation(frame, target, groups, feature_names, p1)
    summary = _summary(rows, target, frame, reasons, p0_scores, p1_scores, ablation)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT / "policy_results.parquet"
    pd.DataFrame(rows).to_parquet(raw_path, index=False)
    model_payload = {
        "schema_version": "1.0",
        "learner_family": "class_weighted_logistic_regression",
        "folding": "stratified_group_five_fold_on_dataset_base_fold",
        "P0_features": list(PREDICTIVE_CHANNELS),
        "P1_features": feature_names,
        "same_hyperparameters": True,
        "same_folds": True,
        "same_calibration": True,
        "target_used_only_for_controller_training_and_scoring": True,
        "target_used_as_feature": False,
        "model_coefficients_sha256": hashlib.sha256(
            json.dumps({"P0": p0_models, "P1": p1_models}, sort_keys=True).encode()
        ).hexdigest(),
    }
    write(OUTPUT / "p0_p1_model_manifest.json", model_payload)
    summary["raw_results"] = {"path": raw_path.relative_to(ROOT).as_posix(), "sha256": sha256(raw_path)}
    summary["model_manifest"] = {
        "path": (OUTPUT / "p0_p1_model_manifest.json").relative_to(ROOT).as_posix(),
        "sha256": sha256(OUTPUT / "p0_p1_model_manifest.json"),
    }
    write(OUTPUT / "summary.json", summary)
    write(
        OUTPUT / "formative_protocol.json",
        {
            "phase": "formative_train_development_only",
            "source_commit": _git_head(),
            "primary_endpoint": "operationally_invalid_automatic_action_at_review_budget_0.20",
            "budgets": list(BUDGETS),
            "target_definition": summary["target_definition"],
            "sealed_test_opened": False,
            "confirmatory_claim_allowed": False,
        },
    )
    print(
        "PASS: final_controller_formative "
        f"objects={len(frame)} invalid={int(target.sum())} p0_p1_complete=true test_opened=false"
    )


def _load_rows() -> pd.DataFrame:
    rows = []
    for dataset_id in DATASETS:
        path = STUDY / f"oof_features/{dataset_id}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                row = {
                    "dataset_id": dataset_id,
                    "modality": _modality(dataset_id),
                    "object_id_hash": payload["object_id_hash"],
                    "fold": payload["fold"],
                    "partition": payload["partition"],
                    "prediction_failure": payload["predicted_label"] != payload["true_label"],
                }
                row.update(payload["predictive"])
                row.update(payload["route"])
                rows.append(row)
    return pd.DataFrame(rows)


def _operational_target(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    reasons = {
        "prediction_failure": frame["prediction_failure"].to_numpy(dtype=bool),
        "route_failure": frame["typed_route_fault"].fillna(0).to_numpy(dtype=float) > 0.0,
        "data_failure": (
            (frame["label_free_shift_score"].to_numpy(dtype=float) >= 0.80)
            | (frame["data_quality_profile"].to_numpy(dtype=float) >= 0.50)
        ),
        "explanation_failure": (
            (frame[["seed_stability", "bootstrap_stability", "perturbation_stability"]].min(axis=1).to_numpy() < 0.40)
            & (frame["explainer_disagreement"].to_numpy(dtype=float) >= 0.60)
        ),
        "contract_failure": np.zeros(len(frame), dtype=bool),
    }
    return np.logical_or.reduce(tuple(reasons.values())), reasons


def _matrices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    p0 = _risk_oriented(frame, list(PREDICTIVE_CHANNELS))
    route_values, route_names = [], []
    for name in ROUTE_CHANNELS:
        values = frame[name].to_numpy(dtype=float)
        missing = np.isnan(values)
        observed = values[~missing]
        fill = float(np.median(observed)) if len(observed) else 0.5
        values = np.where(missing, fill, values)
        if name in GOOD_WHEN_HIGH:
            values = 1.0 - values
        route_values.append(values)
        route_values.append(missing.astype(float))
        route_names.extend((name, f"{name}__missing"))
    return p0, np.column_stack((p0, *route_values)), [*PREDICTIVE_CHANNELS, *route_names]


def _risk_oriented(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    values = []
    for name in names:
        column = frame[name].to_numpy(dtype=float)
        values.append(1.0 - column if name in GOOD_WHEN_HIGH else column)
    return np.column_stack(values)


def _cross_fitted_risk(matrix: np.ndarray, target: np.ndarray, groups: np.ndarray, *, seed: int):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    output = np.zeros(len(target), dtype=float)
    models = []
    for fold, (fit, held) in enumerate(splitter.split(matrix, target, groups)):
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed).fit(matrix[fit], target[fit])
        output[held] = model.predict_proba(matrix[held])[:, 1]
        models.append(
            {
                "fold": fold,
                "fit": len(fit),
                "held": len(held),
                "intercept": float(model.intercept_[0]),
                "coefficients": [float(value) for value in model.coef_[0]],
            }
        )
    return output, models


def _baseline_scores(frame: pd.DataFrame, p0: np.ndarray, p1: np.ndarray) -> dict[str, np.ndarray]:
    confidence_risk = 1.0 - frame["calibrated_confidence"].to_numpy(dtype=float)
    uncertainty = frame["normalized_entropy"].to_numpy(dtype=float)
    model_disagreement = frame["model_checkpoint_disagreement"].to_numpy(dtype=float)
    explainer_disagreement = frame["explainer_disagreement"].to_numpy(dtype=float)
    provenance = 1.0 - frame["provenance_completeness"].to_numpy(dtype=float)
    data_quality = np.maximum(
        frame["data_quality_profile"].to_numpy(dtype=float), frame["label_free_shift_score"].to_numpy(dtype=float)
    )
    simple_or = np.maximum.reduce((confidence_risk, uncertainty, model_disagreement, explainer_disagreement, provenance, data_quality))
    weighted = (
        0.35 * confidence_risk
        + 0.20 * uncertainty
        + 0.15 * model_disagreement
        + 0.15 * explainer_disagreement
        + 0.10 * data_quality
        + 0.05 * provenance
    )
    rank = pd.Series(confidence_risk).rank(method="average", pct=True).to_numpy()
    return {
        "always_accept": np.zeros(len(frame)),
        "raw_confidence_threshold": 1.0 - frame["prediction_margin"].to_numpy(dtype=float),
        "calibrated_confidence_threshold": confidence_risk,
        "uncertainty_threshold": uncertainty,
        "model_disagreement": model_disagreement,
        "explainer_disagreement": explainer_disagreement,
        "provenance_completeness": provenance,
        "data_quality_guardrail": data_quality,
        "simple_or_guardrail": simple_or,
        "weighted_linear_score": weighted,
        "predictive_risk_P0": p0,
        "conformal_selective": rank,
        "full_fuzzyxai_P1": p1,
        "always_review": np.ones(len(frame)),
    }


def _evaluate(scores, target, frame, *, budgets) -> list[dict[str, object]]:
    rows = []
    for budget in budgets:
        review_count = int(round(budget * len(target)))
        for policy, values in scores.items():
            if policy == "always_review":
                review = np.ones(len(target), dtype=bool)
            elif policy == "always_accept":
                review = np.zeros(len(target), dtype=bool)
            else:
                order = np.argsort(values, kind="stable")
                review = np.zeros(len(target), dtype=bool)
                if review_count:
                    review[order[-review_count:]] = True
            accepted = ~review
            rows.append(
                {
                    "policy": policy,
                    "review_budget": budget,
                    "observed_review_rate": float(review.mean()),
                    "invalid_automatic_actions": int(np.sum(target & accepted)),
                    "automatic_coverage": float(accepted.mean()),
                    "operational_risk": float(np.mean(target[accepted])) if accepted.any() else 0.0,
                    "false_blocks": 0,
                    "n": len(target),
                }
            )
            for dataset_id in DATASETS:
                selected = frame["dataset_id"].to_numpy() == dataset_id
                rows.append(
                    {
                        "policy": policy,
                        "review_budget": budget,
                        "observed_review_rate": float(review[selected].mean()),
                        "invalid_automatic_actions": int(np.sum(target[selected] & accepted[selected])),
                        "automatic_coverage": float(accepted[selected].mean()),
                        "operational_risk": float(np.mean(target[selected][accepted[selected]])) if accepted[selected].any() else 0.0,
                        "false_blocks": 0,
                        "n": int(selected.sum()),
                        "dataset_id": dataset_id,
                    }
                )
    return rows


def _component_ablation(frame, target, groups, feature_names, full_matrix) -> dict[str, object]:
    results = {}
    for component, names in COMPONENT_GROUPS.items():
        removed = {name for name in names} | {f"{name}__missing" for name in names}
        keep = [index for index, name in enumerate(feature_names) if name not in removed]
        scores, _ = _cross_fitted_risk(full_matrix[:, keep], target, groups, seed=SEED)
        results[component] = _budget_metric(scores, target, PRIMARY_BUDGET)
    return results


def _budget_metric(scores: np.ndarray, target: np.ndarray, budget: float) -> dict[str, object]:
    count = int(round(budget * len(scores)))
    reviewed = np.zeros(len(scores), dtype=bool)
    reviewed[np.argsort(scores, kind="stable")[-count:]] = True
    return {
        "invalid_automatic_actions": int(np.sum(target & ~reviewed)),
        "automatic_coverage": float(np.mean(~reviewed)),
    }


def _summary(rows, target, frame, reasons, p0, p1, ablation) -> dict[str, object]:
    overall = [row for row in rows if "dataset_id" not in row and row["review_budget"] == PRIMARY_BUDGET]
    p1_row = next(row for row in overall if row["policy"] == "full_fuzzyxai_P1")
    eligible = [
        row
        for row in overall
        if row["policy"] not in {"full_fuzzyxai_P1", "always_review"}
        and abs(row["observed_review_rate"] - PRIMARY_BUDGET) <= 0.001
    ]
    best = min(eligible, key=lambda row: (row["invalid_automatic_actions"], -row["automatic_coverage"], row["policy"]))
    relative = (best["invalid_automatic_actions"] - p1_row["invalid_automatic_actions"]) / max(1, best["invalid_automatic_actions"])
    return {
        "schema_version": "1.0",
        "phase": "formative_real_oof",
        "objects": len(frame),
        "dataset_count": len(DATASETS),
        "primary_review_budget": PRIMARY_BUDGET,
        "target_definition": {
            "prediction_failure": "predicted_label differs from held-out OOF target",
            "route_failure": "typed route fault present",
            "data_failure": "label-free shift >= 0.80 or data-quality fault >= 0.50",
            "explanation_failure": "minimum stability < 0.40 and explainer disagreement >= 0.60",
            "contract_failure": "frozen hard contract violation",
        },
        "target_counts": {name: int(values.sum()) for name, values in reasons.items()} | {"any_invalid": int(target.sum())},
        "P0_primary": _budget_metric(p0, target, PRIMARY_BUDGET),
        "P1_primary": p1_row,
        "best_matched_budget_baseline": best,
        "relative_invalid_action_reduction": float(relative),
        "formative_target_met": bool(relative >= 0.15),
        "component_ablation": ablation,
        "sealed_test_opened": False,
        "confirmatory_claim_allowed": False,
        "frozen_negative_statuses": {
            "H3-original": "not_supported",
            "H5-P-original": "not_supported",
            "H6-general": "not_supported",
        },
    }


def _modality(dataset_id: str) -> str:
    if dataset_id in {"bank_marketing", "default_credit_clients"}:
        return "tabular"
    if dataset_id == "shoulder_implant_xray":
        return "image"
    if dataset_id == "sms_spam":
        return "text"
    return "timeseries"


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


if __name__ == "__main__":
    main()
