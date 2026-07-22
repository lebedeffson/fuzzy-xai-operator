from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np

from baselines.h10 import (
    AnomalyDetectorBaseline,
    HashVersionBaseline,
    IndependentRulesBaseline,
    SchemaOnlyBaseline,
    SimpleOrBaseline,
    TypedRouteBaseline,
    UntypedGraphBaseline,
)
from fuzzyxai.audit_h10 import H10Auditor

from .common import ROOT, load_yaml, write_csv, write_json
from .metrics import evaluate_method, summarize
from .mutations import make_cases
from .routes import build_route


def _routes(prefix: str, count: int) -> list:
    modalities = ("tabular", "text", "time_series")
    return [build_route(f"v17_{modalities[index % 3]}", modalities[index % 3], f"{prefix}-{index:06d}") for index in range(count)]


def _unknown_f1(rows: list[dict]) -> float:
    tp = sum(row["truth_unknown"] and row["predicted_unknown"] for row in rows)
    fp = sum(not row["truth_unknown"] and row["predicted_unknown"] for row in rows)
    fn = sum(row["truth_unknown"] and not row["predicted_unknown"] for row in rows)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def run(config_path: Path) -> None:
    config = load_yaml(config_path)
    out = ROOT / "artifacts" / "h10_v19_exploratory"
    seed = int(config["seed"])
    known = tuple(config["known_leaves"])
    held_out = tuple(config["held_out_leaves"])
    train_routes = _routes("train", int(config["samples_per_modality"]) * 3)
    train_cases = make_cases(train_routes, seed=seed, known_leaves=known, held_out_leaves=(), include_valid=False)
    fit_samples = [(route, truth.leaf_types[0]) for route, truth in train_cases if len(truth.leaf_types) == 1]
    development_routes = _routes("development", int(config["samples_per_modality"]) * 3)
    development_cases = make_cases(development_routes, seed=seed + 1, known_leaves=known, held_out_leaves=held_out)

    tuning_rows = []
    selected = None
    for known_threshold, anomaly_threshold, leaf_threshold in itertools.product(
        config["threshold_grid"]["known"], config["threshold_grid"]["anomaly"], config["threshold_grid"]["leaf"]
    ):
        auditor = H10Auditor.create(
            threshold_known=float(known_threshold), threshold_anomaly=float(anomaly_threshold), leaf_threshold=float(leaf_threshold)
        ).fit(fit_samples)
        rows = evaluate_method("full_h10", auditor.diagnose, development_cases)
        summary = summarize(rows)
        unknown_f1 = _unknown_f1(rows)
        known_rows = [row for row in rows if not row["truth_unknown"] and row["truth_status"] != "valid"]
        known_leaf_accuracy = float(np.mean([row["leaf_correct"] for row in known_rows])) if known_rows else 0.0
        objective = unknown_f1 + 0.25 * known_leaf_accuracy
        item = {
            "threshold_known": known_threshold,
            "threshold_anomaly": anomaly_threshold,
            "leaf_threshold": leaf_threshold,
            "unknown_f1": unknown_f1,
            "known_leaf_accuracy": known_leaf_accuracy,
            "source_f1": summary["source_f1"],
            "repair_f1": summary["repair_f1"],
            "objective": objective,
        }
        tuning_rows.append(item)
        key = (objective, known_leaf_accuracy, -float(known_threshold), -float(anomaly_threshold), -float(leaf_threshold))
        if selected is None or key > selected[0]:
            selected = (key, item, auditor, rows)
    assert selected is not None
    _, selected_thresholds, auditor, full_rows = selected

    methods = {
        "schema_only": SchemaOnlyBaseline().diagnose,
        "hash_version": HashVersionBaseline().diagnose,
        "simple_or": SimpleOrBaseline().diagnose,
        "independent_if_else": IndependentRulesBaseline().diagnose,
        "untyped_graph": UntypedGraphBaseline().diagnose,
        "anomaly_detector": AnomalyDetectorBaseline().diagnose,
        "typed_route": TypedRouteBaseline().diagnose,
        "full_h10": auditor.diagnose,
    }
    all_rows = []
    summaries = []
    for name, method in methods.items():
        rows = full_rows if name == "full_h10" else evaluate_method(name, method, development_cases)
        all_rows.extend(rows)
        summaries.append({"method": name, **summarize(rows)})
    baseline_summaries = [row for row in summaries if row["method"] != "full_h10"]
    best = max(baseline_summaries, key=lambda row: ((row["source_f1"] + row["repair_f1"]) / 2.0, row["method"]))
    full = next(row for row in summaries if row["method"] == "full_h10")
    dataset_effects = []
    for dataset in sorted({row["dataset"] for row in all_rows}):
        for metric in ("source_f1", "repair_f1"):
            left = np.mean([row[metric] for row in all_rows if row["method"] == "full_h10" and row["dataset"] == dataset])
            right = np.mean([row[metric] for row in all_rows if row["method"] == best["method"] and row["dataset"] == dataset])
            dataset_effects.append({"dataset": dataset, "metric": metric, "effect": float(left - right)})
    observed_sd = float(np.std([item["effect"] for item in dataset_effects], ddof=1)) if len(dataset_effects) > 1 else 0.0
    practical_margin = max(0.02, round(0.20 * observed_sd, 4))
    # Normal approximation is a planning aid only; confirmatory inference uses hierarchical bootstrap.
    estimated_datasets_for_80pct = max(3, math.ceil((2.8 * max(observed_sd, 0.01) / practical_margin) ** 2))
    power_analysis = {
        "status": "exploratory_planning_only",
        "best_baseline": best["method"],
        "dataset_effect_sd": observed_sd,
        "practically_relevant_margin": practical_margin,
        "margin_basis": "max(0.02, 0.20 * exploratory between-dataset effect SD)",
        "estimated_datasets_for_80pct_normal_approximation": estimated_datasets_for_80pct,
        "confirmatory_minimum_datasets": 3,
        "warning": "The power approximation does not guarantee a positive result and is not a confirmatory test.",
    }
    write_csv(out / "threshold_tuning.csv", tuning_rows)
    write_csv(out / "development_results.csv", all_rows)
    write_csv(out / "development_summary.csv", summaries)
    write_json(out / "selected_configuration.json", selected_thresholds)
    write_json(out / "power_analysis.json", power_analysis)
    write_json(
        out / "exploratory_summary.json",
        {
            "study_id": config["study_id"],
            "status": "exploratory_not_confirmatory",
            "cases": len(development_cases),
            "selected_configuration": selected_thresholds,
            "best_independent_baseline": best["method"],
            "full_h10": full,
            "best_baseline": best,
            "primary_effects": {
                "source_f1": full["source_f1"] - best["source_f1"],
                "repair_f1": full["repair_f1"] - best["repair_f1"],
            },
            "power_analysis": power_analysis,
            "claim_allowed": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "h10_v19_exploratory.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
