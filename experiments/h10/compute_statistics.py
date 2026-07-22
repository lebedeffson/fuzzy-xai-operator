from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import ARTIFACT_ROOT, ROOT, load_yaml, read_json, write_csv, write_json


def _paired_dataset_arrays(frame: pd.DataFrame, metric: str, baseline: str) -> dict[str, np.ndarray]:
    # Primary localization and repair endpoints are defined on known invalid
    # faults. Unknown faults are evaluated through H10-U abstention metrics.
    subset = frame[(frame["truth_status"] != "valid") & (~frame["truth_unknown"].astype(bool))]
    pivot = subset[subset["method"].isin(("full_h10", baseline))].pivot(
        index=["dataset", "case_id"], columns="method", values=metric
    )
    pivot = pivot.dropna()
    return {
        str(dataset): (group["full_h10"].to_numpy(float) - group[baseline].to_numpy(float))
        for dataset, group in pivot.groupby(level="dataset")
    }


def hierarchical_test(
    frame: pd.DataFrame,
    metric: str,
    baseline: str,
    *,
    repetitions: int,
    seed: int,
    margin: float,
) -> dict:
    arrays = _paired_dataset_arrays(frame, metric, baseline)
    if not arrays:
        raise RuntimeError(f"no paired arrays for {metric} against {baseline}")
    datasets = tuple(sorted(arrays))
    observed_by_dataset = {dataset: float(np.mean(values)) for dataset, values in arrays.items()}
    observed = float(np.mean(list(observed_by_dataset.values())))
    rng = np.random.default_rng(seed)
    boot = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        selected = rng.choice(datasets, size=len(datasets), replace=True)
        effects = []
        for dataset in selected:
            values = arrays[str(dataset)]
            effects.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        boot[index] = float(np.mean(effects))
    low, high = np.quantile(boot, (0.025, 0.975))
    p_raw = (1.0 + float(np.sum(boot <= margin))) / (repetitions + 1.0)
    return {
        "metric": metric,
        "baseline": baseline,
        "effect": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_raw": p_raw,
        "margin": margin,
        "repetitions": repetitions,
        "unit": "dataset_then_object_known_invalid_faults",
        "dataset_effects": observed_by_dataset,
        "direction_consistent": all(value > 0.0 for value in observed_by_dataset.values()),
    }


def _holm(rows: list[dict]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["p_raw"])
    running = 0.0
    total = len(rows)
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, row["p_raw"] * (total - rank))
        running = max(running, adjusted)
        rows[index]["p_holm"] = running


def _unknown_scores(invalid: pd.DataFrame) -> tuple[float, float, float]:
    unknown_truth = invalid["truth_unknown"].astype(bool).to_numpy()
    unknown_score = invalid["anomaly_score"].to_numpy(float)
    if len(np.unique(unknown_truth)) == 2:
        auroc = float(roc_auc_score(unknown_truth, unknown_score))
        auprc = float(average_precision_score(unknown_truth, unknown_score))
    else:
        auroc = float("nan")
        auprc = float("nan")
    unknown_rows = invalid[invalid["truth_unknown"].astype(bool)]
    recall = float(unknown_rows["predicted_unknown"].mean()) if len(unknown_rows) else float("nan")
    return auroc, auprc, recall


def compute(config_path: Path) -> None:
    config = load_yaml(config_path)
    frame = pd.read_csv(ARTIFACT_ROOT / "confirmatory" / "raw_results.csv")
    baseline = config["best_baseline_selected_on_exploratory"]
    bootstrap = config["bootstrap"]
    tests = [
        hierarchical_test(
            frame,
            "source_f1",
            baseline,
            repetitions=int(bootstrap["repetitions"]),
            seed=int(bootstrap["seed"]),
            margin=float(config["practically_relevant_margins"]["source_localization_macro_f1"]),
        ),
        hierarchical_test(
            frame,
            "repair_f1",
            baseline,
            repetitions=int(bootstrap["repetitions"]),
            seed=int(bootstrap["seed"]) + 1,
            margin=float(config["practically_relevant_margins"]["repair_set_f1"]),
        ),
    ]
    _holm(tests)
    for row in tests:
        row["status"] = (
            "supported"
            if row["ci_low"] > row["margin"] and row["p_holm"] < 0.05 and row["direction_consistent"]
            else "not_supported"
        )
    method_rows = []
    for method, group in frame.groupby("method"):
        invalid = group[group["truth_status"] != "valid"]
        known = invalid[~invalid["truth_unknown"].astype(bool)]
        auroc, auprc, unknown_recall = _unknown_scores(invalid)
        method_rows.append(
            {
                "method": method,
                "source_localization_f1": float(known["source_f1"].mean()),
                "repair_set_f1": float(known["repair_f1"].mean()),
                "false_certification": float(group["false_certification"].mean()),
                "false_block": float(group["false_block"].mean()),
                "diagnostic_latency": float(group["diagnostic_latency_ms"].median()),
                "parent_f1": float(known["parent_correct"].mean()),
                "leaf_f1": float(known["leaf_correct"].mean()),
                "unknown_recall": unknown_recall,
                "unknown_auroc": auroc,
                "unknown_auprc": auprc,
                "abstention_accuracy": float(invalid["unknown_correct"].mean()),
                "abstention_rate": float(invalid["abstained"].mean()),
                "cut_exact": float(invalid["cut_exact"].mean()),
                "cut_jaccard": float(invalid["cut_jaccard"].mean()),
                "cut_cost_ratio": float(invalid["cut_cost_ratio"].mean()),
                "extra_nodes": float(invalid["cut_extra_nodes"].mean()),
                "runtime_ms": float(group["diagnostic_latency_ms"].median()),
            }
        )
    dataset_rows = []
    for (dataset, method), group in frame.groupby(["dataset", "method"]):
        invalid = group[group["truth_status"] != "valid"]
        known = invalid[~invalid["truth_unknown"].astype(bool)]
        dataset_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "source_localization_f1": float(known["source_f1"].mean()),
                "repair_set_f1": float(known["repair_f1"].mean()),
                "false_certification": float(group["false_certification"].mean()),
                "false_block": float(group["false_block"].mean()),
            }
        )
    run_summary = json.loads((ARTIFACT_ROOT / "confirmatory" / "run_summary.json").read_text())
    methodology_path = ARTIFACT_ROOT / "closure" / "confirmatory_methodology_audit.json"
    methodology = read_json(methodology_path) if methodology_path.exists() else {"status": "NOT_RUN"}
    if methodology["status"] != "PASS":
        for row in tests:
            row["statistical_status_before_methodology_audit"] = row["status"]
            row["status"] = "invalid_methodology"
    claims = {
        "H10-L": next(row["status"] for row in tests if row["metric"] == "source_f1"),
        "H10-C": "secondary_descriptive_valid_oracle",
        "H10-R": next(row["status"] for row in tests if row["metric"] == "repair_f1"),
        "H10-U": "secondary_descriptive",
        "H10-T": "supported" if run_summary["byte_identical_trace_rate"] == 1.0 else "not_supported",
    }
    full = frame[frame["method"] == "full_h10"]
    safety = {
        "false_certification_rate": float(full["false_certification"].mean()),
        "false_block_rate": float(full["false_block"].mean()),
        "false_certification_boundary": float(config["safety_constraints"]["false_certification_rate"]),
        "false_block_boundary": float(config["safety_constraints"]["false_block_rate"]),
    }
    safety["boundaries_met_point_estimate"] = (
        safety["false_certification_rate"] <= safety["false_certification_boundary"]
        and safety["false_block_rate"] <= safety["false_block_boundary"]
    )
    write_json(ARTIFACT_ROOT / "confirmatory" / "statistical_tests.json", tests)
    write_csv(ARTIFACT_ROOT / "confirmatory" / "method_summary.csv", method_rows)
    write_csv(ARTIFACT_ROOT / "confirmatory" / "dataset_summary.csv", dataset_rows)
    write_json(
        ARTIFACT_ROOT / "closure" / "h10_v19_claim_registry.json",
        {
            "study_id": config["study_id"],
            "claims": claims,
            "primary_tests": tests,
            "safety": safety,
            "methodology_audit_status": methodology["status"],
            "scientific_release_allowed": methodology["status"] == "PASS",
            "identity_anchor_reuse_disclosed": True,
            "old_claims_changed": False,
            "manual_positive_override": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "h10_v19_protocol.yaml")
    args = parser.parse_args()
    compute(args.config)


if __name__ == "__main__":
    main()
