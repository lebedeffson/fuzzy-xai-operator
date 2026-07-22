from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .common import ARTIFACT_ROOT, ROOT, load_config, write_json


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _paired_by_pipeline(rows: list[dict[str, str]], metric: str, baseline: str) -> dict[str, np.ndarray]:
    selected = [
        row for row in rows
        if row["case_type"] == "composite"
        and row["unknown"].lower() == "false"
        and row["method"] in {"full_h10", baseline}
    ]
    values = {(row["pipeline_id"], row["case_id"], row["method"]): float(row[metric]) for row in selected}
    result: dict[str, list[float]] = {}
    for pipeline, case_id, method in values:
        if method != "full_h10":
            continue
        key_baseline = (pipeline, case_id, baseline)
        if key_baseline in values:
            result.setdefault(pipeline, []).append(values[(pipeline, case_id, "full_h10")] - values[key_baseline])
    return {pipeline: np.asarray(items, dtype=float) for pipeline, items in result.items()}


def hierarchical_test(
    rows: list[dict[str, str]], metric: str, baseline: str, repetitions: int, seed: int, margin: float
) -> dict[str, Any]:
    arrays = _paired_by_pipeline(rows, metric, baseline)
    pipelines = tuple(sorted(arrays))
    effects_by_pipeline = {pipeline: float(values.mean()) for pipeline, values in arrays.items()}
    observed = float(np.mean(list(effects_by_pipeline.values())))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled_pipelines = rng.choice(pipelines, size=len(pipelines), replace=True)
        pipeline_effects = []
        for pipeline in sampled_pipelines:
            values = arrays[str(pipeline)]
            pipeline_effects.append(float(rng.choice(values, size=len(values), replace=True).mean()))
        bootstrap[index] = float(np.mean(pipeline_effects))
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    p_raw = (1.0 + float(np.sum(bootstrap <= 0.0))) / (repetitions + 1.0)
    return {
        "metric": metric,
        "baseline": baseline,
        "effect": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_raw": p_raw,
        "registered_margin": margin,
        "pipeline_effects": effects_by_pipeline,
        "positive_pipeline_count": sum(value > 0.0 for value in effects_by_pipeline.values()),
        "pipeline_count": len(effects_by_pipeline),
        "bootstrap_repetitions": repetitions,
        "unit": "pipeline_then_paired_case",
    }


def _holm(tests: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(tests), key=lambda item: item[1]["p_raw"])
    running = 0.0
    for rank, (index, row) in enumerate(ordered):
        running = max(running, min(1.0, row["p_raw"] * (len(tests) - rank)))
        tests[index]["p_holm"] = running


def compute(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    rows = _read(ARTIFACT_ROOT / "confirmatory" / "sealed_test_raw_results.csv")
    lock = __import__("json").loads((ARTIFACT_ROOT / "lock" / "protocol_lock.json").read_text())
    baseline = lock["best_baseline_selected_on_development"]
    bootstrap = config["bootstrap"]
    tests = [
        hierarchical_test(
            rows,
            "source_f1",
            baseline,
            int(bootstrap["repetitions"]),
            int(bootstrap["seed"]),
            float(config["registered_margins"]["source_localization_macro_f1"]),
        ),
        hierarchical_test(
            rows,
            "repair_f1",
            baseline,
            int(bootstrap["repetitions"]),
            int(bootstrap["seed"]) + 1,
            float(config["registered_margins"]["repair_set_macro_f1"]),
        ),
    ]
    _holm(tests)
    full_rows = [row for row in rows if row["method"] == "full_h10"]
    false_certification = float(np.mean([float(row["false_certification"]) for row in full_rows]))
    false_block = float(np.mean([float(row["false_block"]) for row in full_rows]))
    safety_met = (
        false_certification <= float(config["safety_constraints"]["false_certification"])
        and false_block <= float(config["safety_constraints"]["false_block"])
    )
    for test in tests:
        test["status"] = "supported" if (
            test["effect"] >= test["registered_margin"]
            and test["ci_low"] > 0.0
            and test["p_holm"] < 0.05
            and test["positive_pipeline_count"] >= 5
            and safety_met
        ) else "not_supported"
    registry = {
        "study_id": config["study_id"],
        "primary_population": "composite_known_only",
        "claims": {"H10-L": tests[0]["status"], "H10-R": tests[1]["status"], "H10-C": "secondary", "H10-U": "secondary"},
        "primary_tests": tests,
        "safety": {"false_certification": false_certification, "false_block": false_block, "constraints_met": safety_met},
        "old_v16_changed": False,
        "old_v18_changed": False,
        "old_v19_changed": False,
        "manual_positive_override": False,
    }
    write_json(ARTIFACT_ROOT / "confirmatory" / "h10_final_gold_statistics.json", tests)
    write_json(ARTIFACT_ROOT / "closure" / "h10_final_gold_claim_registry.json", registry)
    return registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "h10_final_gold_protocol.yaml")
    args = parser.parse_args()
    compute(args.config)


if __name__ == "__main__":
    main()
