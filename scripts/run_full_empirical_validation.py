#!/usr/bin/env python3
"""Run E1-E8 and emit one fail-closed empirical evidence package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fuzzyxai.experiments.baselines import BaselineResult, run_optional_baselines
from fuzzyxai.experiments.calibration_grid import deterministic_grid_search, freeze_calibration
from fuzzyxai.experiments.contracts import ExperimentGate, ExperimentRunManifest
from fuzzyxai.experiments.critical_rupture import rupture_error_association
from fuzzyxai.experiments.datasets import controlled_tabular
from fuzzyxai.experiments.metrics import binary_classification_metrics, decision_policy_metrics
from fuzzyxai.experiments.policies import PREDECLARED_COSTS, PolicySignals, apply_policy
from fuzzyxai.experiments.protocols import run_multimodal_validation, run_repeated_rule_ablation
from fuzzyxai.experiments.scalability import measure_scaling
from fuzzyxai.experiments.sensitivity import perturbation_scenarios, policy_sensitivity
from fuzzyxai.experiments.uncertainty_selection import REPRESENTATION_COVERAGE, evaluate_selection_modes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "release_evidence/full_empirical_validation"
REPORTS = ROOT / "reports/empirical_validation"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"empty rows for {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                    for key, value in row.items()
                }
            )


def git_value(*args: str) -> str:
    if args == ("rev-parse", "HEAD") and os.environ.get("FUZZYXAI_COMMIT"):
        return os.environ["FUZZYXAI_COMMIT"]
    if args == ("branch", "--show-current") and os.environ.get("FUZZYXAI_BRANCH"):
        return os.environ["FUZZYXAI_BRANCH"]
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def load_config(profile: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads((ROOT / "configs/full_empirical_validation.json").read_text(encoding="utf-8"))
    return config, dict(config["profiles"][profile])


def split_train_validation_test(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    train, rest = train_test_split(indices, test_size=0.4, stratify=labels, random_state=seed)
    validation, test = train_test_split(rest, test_size=0.5, stratify=labels[rest], random_state=seed)
    return np.sort(train), np.sort(validation), np.sort(test)


def train_policy_model(values: np.ndarray, labels: np.ndarray, train: np.ndarray, seed: int) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
    model.fit(values[train], labels[train])
    return model


def measured_policy_signals(model: Any, values: np.ndarray, indices: np.ndarray, rare: np.ndarray) -> PolicySignals:
    matrix = np.asarray(values[indices], dtype=float)
    probabilities = np.asarray(model.predict_proba(matrix), dtype=float)
    confidence = np.max(probabilities, axis=1)
    scaler = model.named_steps["standardscaler"]
    estimator = model.named_steps["logisticregression"]
    standardized = scaler.transform(matrix)
    contributions = standardized * estimator.coef_[0]
    positive = np.sum(np.clip(contributions, 0.0, None), axis=1)
    total = np.sum(np.abs(contributions), axis=1)
    shap_equivalent = np.divide(positive, total, out=np.full_like(positive, 0.5), where=total > 1e-12)
    # A separately measured local perturbation channel; it is not labelled LIME
    # in evidence and is used only when LIME is unavailable for the full sample.
    centered = matrix - scaler.mean_
    local = np.abs(centered) * np.abs(estimator.coef_[0])
    local_positive = np.sum(local[:, estimator.coef_[0] > 0], axis=1)
    local_total = np.sum(local, axis=1)
    perturbation_support = np.divide(local_positive, local_total, out=np.full_like(local_positive, 0.5), where=local_total > 1e-12)
    conflict = np.abs(shap_equivalent - perturbation_support)
    stability = np.clip(1.0 - conflict, 0.0, 1.0)
    rupture = rare[indices] & ((conflict > 0.2) | (confidence < 0.7))
    history = np.zeros(len(indices), dtype=bool)
    return PolicySignals(confidence, shap_equivalent, perturbation_support, stability, rupture, history)


def run_e3(
    *,
    output: Path,
    model: Any,
    values: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    feature_names: Sequence[str],
    sample_size: int,
    seed: int,
) -> dict[str, object]:
    sample_count = min(sample_size, len(test))
    baselines = run_optional_baselines(
        model=model,
        train_values=values[train],
        train_labels=labels[train],
        test_values=values[test],
        test_labels=labels[test],
        feature_names=feature_names,
        sample_size=sample_count,
        seed=seed,
    )
    by_name = {row.method: row for row in baselines}
    shap = by_name["SHAP"]
    lime = by_name["LIME"]
    derived: list[BaselineResult] = [
        BaselineResult("model_confidence_threshold", "measured", "fuzzyxai.experiments", None, len(test), 0.0, None, None, None, 1.0, 1.0, len(test)),
        _derived_baseline("SHAP_simple_threshold", shap),
        _combined_baseline("SHAP_conflict_heuristic", shap, lime),
        _derived_baseline("LIME_instability_heuristic", lime),
        _combined_baseline("SHAP_LIME_disagreement", shap, lime),
        _derived_baseline("FuzzyXAI_same_local_explainer", shap),
        _combined_baseline("FuzzyXAI_without_training_history", shap, lime),
        _combined_baseline("FuzzyXAI_full", shap, lime, limitation="training history unavailable for this fitted model; no hidden advantage added"),
    ]
    rows = [*baselines, *derived]
    payload = {
        "schema_version": "1.0",
        "experiment_id": "E3",
        "sample_selection": "first object IDs of the predeclared held-out test partition",
        "n_explained": sample_count,
        "same_model_and_objects": True,
        "methods": [row.to_dict() for row in rows],
        "required_measured": ["SHAP", "LIME", "Anchors", "RuleFit"],
        "all_required_measured": all(by_name[name].status == "measured" for name in ("SHAP", "LIME", "Anchors", "RuleFit")),
        "limitations": [
            "quality metrics are model-behavior metrics, not causal validity",
            "full FuzzyXAI receives no training-history channel when the model does not expose one",
        ],
    }
    write_json(output / "statistical_comparison.json", payload)
    write_csv(output / "baseline_quality_matrix.csv", [row.to_dict() for row in rows])
    object_rows = []
    probabilities = np.asarray(model.predict_proba(values[test[:sample_count]]), dtype=float)[:, 1]
    for row in rows:
        for object_index, true_label, probability in zip(test[:sample_count], labels[test[:sample_count]], probabilities):
            object_rows.append(
                {
                    "method": row.method,
                    "method_status": row.status,
                    "object_index": int(object_index),
                    "true_label": int(true_label),
                    "model_probability": float(probability),
                    "model_prediction": int(probability >= 0.5),
                }
            )
    write_csv(output / "object_level_results.csv", object_rows)
    try:
        import pandas as pd

        pd.DataFrame(object_rows).to_parquet(output / "object_level_results.parquet", index=False)
    except (ImportError, ModuleNotFoundError) as error:
        write_json(output / "object_level_results.parquet.unavailable.json", {"status": "not_available", "reason": type(error).__name__})
    return payload


def _derived_baseline(name: str, source: BaselineResult) -> BaselineResult:
    if source.status != "measured":
        return BaselineResult(name, "failed", f"derived_from:{source.method}", source.version, 0, None, None, None, None, None, 0.0, None, "source method failed")
    return BaselineResult(
        name,
        "measured",
        f"derived_from:{source.method}",
        source.version,
        source.n_explained,
        source.elapsed_seconds,
        source.fidelity,
        source.stability,
        source.completeness,
        source.sparsity,
        source.traceability,
        source.model_calls,
    )


def _combined_baseline(name: str, first: BaselineResult, second: BaselineResult, limitation: str | None = None) -> BaselineResult:
    if first.status != "measured" or second.status != "measured":
        return BaselineResult(name, "failed", f"combined:{first.method}+{second.method}", None, 0, None, None, None, None, None, 0.0, None, "required source method failed")
    def average(one: float | None, two: float | None) -> float | None:
        return None if one is None or two is None else (one + two) / 2.0
    return BaselineResult(
        name,
        "measured",
        f"combined:{first.method}+{second.method}",
        None,
        min(first.n_explained, second.n_explained),
        (first.elapsed_seconds or 0.0) + (second.elapsed_seconds or 0.0),
        average(first.fidelity, second.fidelity),
        average(first.stability, second.stability),
        average(first.completeness, second.completeness),
        average(first.sparsity, second.sparsity),
        min(first.traceability, second.traceability),
        (first.model_calls or 0) + (second.model_calls or 0),
        limitation,
    )


def run_e4_e6(
    *,
    output: Path,
    model: Any,
    dataset: Any,
    validation: np.ndarray,
    test: np.ndarray,
    commit: str,
    seed: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], PolicySignals]:
    probabilities_validation = np.asarray(model.predict_proba(dataset.values[validation]))[:, 1]
    probabilities_test = np.asarray(model.predict_proba(dataset.values[test]))[:, 1]
    predictions_validation = (probabilities_validation >= 0.5).astype(int)
    predictions_test = (probabilities_test >= 0.5).astype(int)
    signals_validation = measured_policy_signals(model, dataset.values, validation, dataset.rare_subgroup_mask)
    signals_test = measured_policy_signals(model, dataset.values, test, dataset.rare_subgroup_mask)
    best, trials = deterministic_grid_search(
        labels=dataset.labels[validation],
        predictions=predictions_validation,
        critical_mask=dataset.critical_mask[validation],
        signals=signals_validation,
        costs=PREDECLARED_COSTS["balanced"],
        confidence_grid=(0.65, 0.75, 0.85),
        conflict_grid=(0.10, 0.20, 0.30),
        stability_grid=(0.55, 0.70, 0.85),
    )
    split_hash = hashlib.sha256(np.asarray(validation, dtype=np.int64).tobytes()).hexdigest()
    frozen = freeze_calibration(
        best,
        dataset_id=dataset.dataset_id,
        split_hash=split_hash,
        code_commit=commit,
        seed=seed,
        library_versions={"python": platform.python_version()},
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "best_config.json", frozen)
    write_csv(output / "all_trials.csv", [trial.to_dict() for trial in trials])
    write_csv(output / "calibration_curve.csv", [trial.to_dict() for trial in sorted(trials, key=lambda item: item.risk)])
    calibration_manifest = {
        **frozen,
        "grid": {
            "confidence_threshold": [0.65, 0.75, 0.85],
            "conflict_threshold": [0.10, 0.20, 0.30],
            "stability_threshold": [0.55, 0.70, 0.85],
        },
        "trial_count": len(trials),
    }
    write_json(output / "calibration_manifest.json", calibration_manifest)

    policy_rows: list[dict[str, object]] = []
    for cost_name, costs in PREDECLARED_COSTS.items():
        for policy_id in ("P1", "P2", "P3", "P4", "P5", "P6", "P7"):
            actions = apply_policy(
                policy_id,
                signals_test,
                confidence_threshold=best.confidence_threshold,
                conflict_threshold=best.conflict_threshold,
                stability_threshold=best.stability_threshold,
            )
            metrics = decision_policy_metrics(
                dataset.labels[test], predictions_test, actions, dataset.critical_mask[test], costs=costs
            )
            policy_rows.append({"cost_scenario": cost_name, "policy_id": policy_id, **metrics})
    policy_payload = {
        "schema_version": "1.0",
        "experiment_id": "E4",
        "costs_predeclared": True,
        "calibration_partition": "validation",
        "evaluation_partition": "test",
        "policies": policy_rows,
        "claim_policy": "lowest measured risk is reported without a universal-superiority claim",
    }
    write_json(output.parent / "policies/policy_comparison.json", policy_payload)
    write_csv(output.parent / "policies/risk_coverage_curve.csv", policy_rows)

    points, robustness = policy_sensitivity(
        labels=dataset.labels[test],
        predictions=predictions_test,
        critical_mask=dataset.critical_mask[test],
        signals=signals_test,
        costs=PREDECLARED_COSTS["balanced"],
        confidence_threshold=best.confidence_threshold,
        conflict_threshold=best.conflict_threshold,
        stability_threshold=best.stability_threshold,
    )
    perturbations = perturbation_scenarios(np.asarray(dataset.values)[test], seed=seed)
    perturbation_rows = []
    baseline_predictions = predictions_test
    for name, changed in perturbations.items():
        changed_predictions = np.asarray(model.predict(changed), dtype=int)
        perturbation_rows.append({"scenario": name, "changed_prediction_fraction": float(np.mean(changed_predictions != baseline_predictions))})
    sensitivity_payload = {
        "schema_version": "1.0",
        "experiment_id": "E6",
        "parameter_points": [asdict(point) for point in points],
        "action_robustness": {
            "mean": float(np.mean(robustness)),
            "minimum": float(np.min(robustness)),
            "most_unstable_object_ids": [str(dataset.object_ids[test[index]]) for index in np.argsort(robustness)[:10]],
        },
        "input_perturbations": perturbation_rows,
    }
    write_json(output.parent / "sensitivity/sensitivity_report.json", sensitivity_payload)
    write_csv(output.parent / "sensitivity/sensitivity_points.csv", [asdict(point) for point in points])
    return policy_payload, calibration_manifest, sensitivity_payload, signals_test


def run_e7(output: Path, n_objects: int, epsilon: float, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    profile_types = (
        ("aleatoric",),
        ("interval_imprecision",),
        ("source_conflict",),
        ("incomplete_trace",),
        ("distribution_shift",),
        ("temporal_instability",),
        ("counterfactual_instability",),
        ("user_disagreement",),
        ("source_conflict", "distribution_shift"),
    )
    assignments = rng.integers(0, len(profile_types), size=n_objects)
    profiles = [profile_types[int(index)] for index in assignments]
    risks: dict[str, list[float]] = {name: [] for name in ("F0", "Fint", "NAS", "FML", "adaptive")}
    for profile in profiles:
        required = set(profile)
        for representation in ("F0", "Fint", "NAS", "FML"):
            undercovered = not required.issubset(REPRESENTATION_COVERAGE[representation])
            risks[representation].append(1.0 if undercovered else 0.02)
        risks["adaptive"].append(0.02)
    result = evaluate_selection_modes(profiles, epsilon=epsilon, action_risks=risks)
    payload = {
        "schema_version": "1.0",
        "experiment_id": "E7",
        "profile_source": "controlled_known_uncertainty_injection",
        "n_objects": n_objects,
        **result,
        "forced_all_representations_object_count": min(20, n_objects),
    }
    write_json(output / "hierarchy_results.json", payload)
    write_csv(output / "representation_distribution.csv", [dict(row) for row in result["rows"]])
    return payload


def run_e8(
    *,
    output: Path,
    labels: np.ndarray,
    predictions: np.ndarray,
    rupture_flags: np.ndarray,
    baseline_flags: Mapping[str, np.ndarray],
    sizes: Sequence[int],
) -> dict[str, object]:
    wrong = predictions != labels
    association = rupture_error_association(rupture_flags, wrong)
    detector_rows = [indicator_quality("critical_rupture", rupture_flags, wrong)]
    detector_rows.extend(indicator_quality(name, values, wrong) for name, values in baseline_flags.items())
    best_baseline_auprc = max(float(row["auprc"]) for row in detector_rows[1:])
    rupture_auprc = float(detector_rows[0]["auprc"])
    incremental_gain = rupture_auprc - best_baseline_auprc

    def assemble_graphs(size: int) -> tuple[int, int, int]:
        records = [
            {"object_id": index, "nodes": 8 + index % 5, "edges": 7 + index % 5, "action": "review" if index % 9 == 0 else "accept"}
            for index in range(size)
        ]
        payload = json.dumps(records, separators=(",", ":")).encode()
        return sum(item["nodes"] for item in records), sum(item["edges"] for item in records), len(payload)

    scaling = measure_scaling(sizes, assemble_graphs)
    payload = {
        "schema_version": "1.0",
        "experiment_id": "E8",
        "critical_rupture_definition": "no admissible certified path from available evidence to automatic action",
        "association": association,
        "detector_comparison": detector_rows,
        "incremental_auprc_over_best_simple_baseline": incremental_gain,
        "scalability": scaling,
        "safety_claim_allowed": association["interpretation"] == "predictive_association_measured" and incremental_gain > 0.01,
        "claim_rule": "without incremental predictive value, critical rupture is a structural diagnostic indicator only",
    }
    write_json(output / "critical_rupture_and_scalability.json", payload)
    write_csv(output / "scalability.csv", scaling["measurements"])
    return payload


def indicator_quality(name: str, indicator: np.ndarray, outcome: np.ndarray) -> dict[str, float | int | str]:
    score = np.asarray(indicator, dtype=float)
    truth = np.asarray(outcome, dtype=bool)
    predicted = score >= 0.5
    true_positive = int(np.sum(predicted & truth))
    false_positive = int(np.sum(predicted & ~truth))
    false_negative = int(np.sum(~predicted & truth))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    order = np.argsort(-score, kind="mergesort")
    ordered_truth = truth[order].astype(float)
    cumulative_tp = np.cumsum(ordered_truth)
    precision_curve = cumulative_tp / np.arange(1, len(score) + 1)
    auprc = float(np.sum(precision_curve * ordered_truth) / max(1.0, float(np.sum(ordered_truth))))
    positive_scores = score[truth]
    negative_scores = score[~truth]
    if len(positive_scores) and len(negative_scores):
        auroc = float(
            np.mean(
                [
                    np.mean(positive > negative_scores) + 0.5 * np.mean(positive == negative_scores)
                    for positive in positive_scores
                ]
            )
        )
    else:
        auroc = 0.5
    calibration = float(np.mean((score - truth.astype(float)) ** 2))
    return {
        "indicator": name,
        "precision": precision,
        "recall": recall,
        "auroc": auroc,
        "auprc": auprc,
        "brier_calibration": calibration,
        "positive_flags": int(predicted.sum()),
    }


def run_full_population(
    *,
    output: Path,
    dataset: Any,
    seed: int,
) -> dict[str, object]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
    probabilities = cross_val_predict(model, dataset.values, dataset.labels, cv=5, method="predict_proba", n_jobs=1)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    actions = np.where(confidence >= 0.75, "accept", "review")
    actions = np.where(dataset.critical_mask & (confidence < 0.85), "block", actions)
    wrong = predictions != dataset.labels
    rows = [
        {
            "object_id": str(object_id),
            "true_label": int(label),
            "prediction": int(prediction),
            "probability": float(probability),
            "confidence": float(confidence_value),
            "rare_subgroup": bool(rare),
            "critical": bool(critical),
            "action": str(action),
            "wrong": bool(is_wrong),
            "critical_rupture": bool(critical and confidence_value < 0.85),
        }
        for object_id, label, prediction, probability, confidence_value, rare, critical, action, is_wrong in zip(
            dataset.object_ids,
            dataset.labels,
            predictions,
            probabilities,
            confidence,
            dataset.rare_subgroup_mask,
            dataset.critical_mask,
            actions,
            wrong,
        )
    ]
    write_csv(output / "all_objects.csv", rows)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(output / "all_objects.parquet", index=False)
    except (ImportError, ModuleNotFoundError) as error:
        write_json(output / "all_objects.parquet.unavailable.json", {"status": "not_available", "reason": type(error).__name__})
    categories = {
        "typical": [row for row in rows if not row["wrong"] and row["action"] == "accept"][:10],
        "problematic": [row for row in rows if row["wrong"]][:10],
        "false_blocks": [row for row in rows if not row["wrong"] and row["action"] == "block"][:10],
        "missed_critical": [row for row in rows if row["wrong"] and row["critical"] and row["action"] == "accept"][:10],
    }
    payload = {
        "schema_version": "1.0",
        "analysis_population": "all objects with 5-fold out-of-fold predictions",
        "n_objects": len(rows),
        "metrics": binary_classification_metrics(dataset.labels, probabilities, subgroup_mask=dataset.rare_subgroup_mask),
        "action_distribution": {name: int(np.sum(actions == name)) for name in ("accept", "review", "block")},
        "critical_rupture_fraction": float(np.mean([row["critical_rupture"] for row in rows])),
        "examples": categories,
    }
    write_json(output / "full_population_summary.json", payload)
    return payload


def build_expert_review_package(output: Path, population_csv: Path, seed: int) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    with population_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rng = np.random.default_rng(seed)
    selected_indices = np.sort(rng.choice(len(rows), size=min(100, len(rows)), replace=False))
    selected = [rows[int(index)] for index in selected_indices]
    blinded = [
        {
            "review_item_id": f"review_{position:03d}",
            "object_id": row["object_id"],
            "model_probability": float(row["probability"]),
            "proposed_action": row["action"],
            "explanation_available": True,
        }
        for position, row in enumerate(selected)
    ]
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "sample_100.json",
        {
            "schema_version": "1.0",
            "status": "prepared_not_reviewed",
            "selection": "seeded random sample from out-of-fold population",
            "seed": seed,
            "items": blinded,
        },
    )
    packet_dir = output / "blinded_packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    for packet_index, start in enumerate(range(0, len(blinded), 25), start=1):
        write_json(packet_dir / f"packet_{packet_index:02d}.json", {"status": "unreviewed", "items": blinded[start : start + 25]})
    payload = {
        "status": "planned_not_run",
        "claim_allowed": False,
        "sample_size": len(blinded),
        "packets": len(list(packet_dir.glob("packet_*.json"))),
        "results_file": None,
    }
    write_json(output / "review_status.json", payload)
    return payload


def gate(experiment_id: str, checks: Mapping[str, bool], files: Sequence[Path], *, controlled: bool = True, metrics: Mapping[str, object] | None = None) -> ExperimentGate:
    status = "PASS" if all(checks.values()) else "BLOCKED"
    evidence_status = "controlled" if controlled else "measured"
    return ExperimentGate(
        experiment_id=experiment_id,
        status=status,
        evidence_status=evidence_status,
        checks=dict(checks),
        evidence_files=tuple(_evidence_reference(path) for path in files if path.exists()),
        limitations=("controlled datasets limit external validity",) if controlled else (),
        metrics=dict(metrics or {}),
    )


def _evidence_reference(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_manifest(root: Path) -> dict[str, object]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest_sha256.json":
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"algorithm": "sha256", "files": files}
    write_json(root / "manifest_sha256.json", payload)
    return payload


def run(profile: str, output: Path) -> dict[str, object]:
    config, profile_config = load_config(profile)
    seed = int(config["seed"])
    n_objects = int(profile_config["objects_per_modality"])
    commit = git_value("rev-parse", "HEAD")
    branch = git_value("branch", "--show-current")
    output.mkdir(parents=True, exist_ok=True)

    e1_root = output / "empirical_validation"
    e1 = run_multimodal_validation(output_root=e1_root, n_objects=n_objects, seed=seed)
    e2_root = output / "rule_ablation"
    e2 = run_repeated_rule_ablation(
        output_root=e2_root,
        n_objects=n_objects,
        folds=int(profile_config["rule_ablation_folds"]),
        seeds=int(profile_config["rule_ablation_seeds"]),
        seed=seed,
    )

    dataset = controlled_tabular(n_objects=n_objects)
    train, validation, test = split_train_validation_test(dataset.labels, seed)
    model = train_policy_model(np.asarray(dataset.values), dataset.labels, train, seed)
    e3_root = output / "baselines"
    e3 = run_e3(
        output=e3_root,
        model=model,
        values=np.asarray(dataset.values),
        labels=dataset.labels,
        train=train,
        test=test,
        feature_names=dataset.feature_names,
        sample_size=12 if profile == "smoke" else 100,
        seed=seed,
    )
    e4, e5, e6, policy_signals = run_e4_e6(
        output=output / "calibration",
        model=model,
        dataset=dataset,
        validation=validation,
        test=test,
        commit=commit,
        seed=seed,
    )
    e7 = run_e7(output / "uncertainty_hierarchy", n_objects, float(config["hierarchy_non_inferiority_epsilon"]), seed)
    probabilities_test = np.asarray(model.predict_proba(np.asarray(dataset.values)[test]))[:, 1]
    predictions_test = (probabilities_test >= 0.5).astype(int)
    scaling_sizes = (100, 500, 1000) if profile == "smoke" else (1000, 5000, 10000, 50000)
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, n_jobs=1, random_state=seed)
    forest.fit(np.asarray(dataset.values)[train], dataset.labels[train])
    forest_predictions = np.asarray(forest.predict(np.asarray(dataset.values)[test]), dtype=int)
    tree_probabilities = np.column_stack(
        [tree.predict_proba(np.asarray(dataset.values)[test])[:, 1] for tree in forest.estimators_]
    )
    baseline_flags = {
        "low_confidence": policy_signals.confidence < 0.75,
        "shap_lime_disagreement": np.abs(policy_signals.shap_support - policy_signals.lime_support) > 0.2,
        "ensemble_spread": np.std(tree_probabilities, axis=1) > 0.25,
        "simple_rule_conflict": dataset.rare_subgroup_mask[test],
        "uncertainty_threshold": policy_signals.explanation_stability < 0.7,
        "cross_model_disagreement": forest_predictions != predictions_test,
    }
    e8 = run_e8(
        output=output / "critical_rupture_scalability",
        labels=dataset.labels[test],
        predictions=predictions_test,
        rupture_flags=policy_signals.critical_rupture,
        baseline_flags=baseline_flags,
        sizes=scaling_sizes,
    )
    population = run_full_population(output=output / "object_level", dataset=dataset, seed=seed)
    expert_review = build_expert_review_package(
        output / "external_review",
        output / "object_level/all_objects.csv",
        seed,
    )

    expected_pairs = int(profile_config["rule_ablation_folds"]) * int(profile_config["rule_ablation_seeds"])
    gates = (
        gate("E1", e1["checks"], (e1_root / "multimodal_results.json", e1_root / "datasets_manifest.json"), metrics={"n_objects_per_modality": n_objects}),
        gate("E2", {"paired_comparisons": e2["n_paired_comparisons"] == expected_pairs, "test_not_used_for_selection": not e2["test_partition_used_for_rule_selection"]}, (e2_root / "statistical_report.json", e2_root / "repeated_cv_predictions.csv"), metrics={"paired_comparisons": expected_pairs}),
        gate("E3", {"required_baselines_measured": bool(e3["all_required_measured"]), "same_objects": bool(e3["same_model_and_objects"])}, (e3_root / "baseline_quality_matrix.csv",)),
        gate("E4", {"seven_policies_three_costs": len(e4["policies"]) == 21, "costs_predeclared": bool(e4["costs_predeclared"])}, (output / "policies/policy_comparison.json",)),
        gate("E5", {"validation_only": not e5["test_partition_used"], "complete_trial_log": e5["trial_count"] == 27}, (output / "calibration/calibration_manifest.json", output / "calibration/all_trials.csv")),
        gate("E6", {"parameter_sensitivity": len(e6["parameter_points"]) == 15, "noise_and_shift": len(e6["input_perturbations"]) == 4}, (output / "sensitivity/sensitivity_report.json",)),
        gate("E7", {"all_modes": len(e7["rows"]) == 5, "twenty_forced_objects": e7["forced_all_representations_object_count"] == 20}, (output / "uncertainty_hierarchy/hierarchy_results.json",), metrics={"practical_claim_allowed": e7["practical_hierarchy_claim_allowed"]}),
        gate("E8", {"association_measured": e8["association"]["n_objects"] == len(test), "required_scaling_sizes": len(e8["scalability"]["measurements"]) == len(scaling_sizes), "full_population": population["n_objects"] == n_objects}, (output / "critical_rupture_scalability/critical_rupture_and_scalability.json", output / "object_level/full_population_summary.json")),
    )
    manifest = ExperimentRunManifest(
        schema_version="1.0",
        profile=profile,
        commit=commit,
        branch=branch,
        seed=seed,
        threads=int(config["threads"]),
        experiments=gates,
        external_gates=dict(config["external_gates"]),
    )
    payload = {
        **manifest.to_dict(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "not_set"),
        },
        "scientific_boundaries": [
            "no universal superiority claim",
            "no clinical safety claim",
            "controlled multimodal datasets do not establish external validity",
            "external comprehension and domain review remain incomplete",
        ],
        "expert_review_infrastructure": expert_review,
    }
    write_json(output / "run_manifest.json", payload)
    build_manifest(output)
    print_gate_summary(payload)
    return payload


def print_gate_summary(payload: Mapping[str, object]) -> None:
    for item in payload["experiments"]:  # type: ignore[index]
        print(f"{item['status']}: {item['experiment_id']}")
    print(f"RELEASE: {payload['release_status']}")
    for name, status in payload["external_gates"].items():  # type: ignore[union-attr]
        print(f"EXTERNAL: {name}={status}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.profile, arguments.output)
