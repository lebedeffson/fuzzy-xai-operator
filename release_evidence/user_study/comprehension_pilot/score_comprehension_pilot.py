"""Score the external Human Explanation comprehension pilot without inventing responses."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


BOOLEAN_FIELDS = (
    "decision_correct",
    "reasons_correct",
    "concern_correct",
    "reliability_correct",
    "action_correct",
    "limitation_correct",
)
EXTENDED_BOOLEAN_FIELDS = (
    "provenance_correct",
    "similarity_correct",
    "counterfactual_correct",
    "native_surrogate_correct",
)
ERROR_BOOLEAN_FIELDS = (
    "overtrust_error",
    "iou_misinterpreted_as_probability",
    "sensitivity_misinterpreted_as_recommendation",
)
MODES = ("technical_baseline", "human_explanation")
REQUIRED_SCENARIOS = {"forgetting_case", "rule_ablation", "image_similarity"}


def _boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"invalid boolean value: {value!r}")
    return normalized in {"true", "1", "yes"}


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return round(sum(items) / len(items), 6) if items else 0.0


def _row_boolean(row: dict[str, str], field: str, default: bool = False) -> bool:
    value = row.get(field, "")
    return default if not value.strip() else _boolean(value)


def score_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {
            "schema_version": "1.0",
            "status": "planned_not_run",
            "participant_count": 0,
            "claim_allowed": False,
            "reason": "No independent participant responses have been recorded.",
        }

    participants = {row["participant_id"].strip() for row in rows if row.get("participant_id", "").strip()}
    if len(participants) < 6:
        raise ValueError("comprehension pilot requires at least six independent participants")
    roles = Counter(row["role"].strip() for row in rows)
    if roles["domain_specialist"] < 3 or roles["model_integrator"] < 3:
        raise ValueError("pilot requires at least three domain specialists and three model integrators")
    modes_by_participant: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        mode = row["mode"].strip()
        if mode not in MODES:
            raise ValueError(f"unsupported comparison mode: {mode}")
        modes_by_participant[row["participant_id"].strip()].add(mode)
    if any(modes != set(MODES) for modes in modes_by_participant.values()):
        raise ValueError("every participant must evaluate both comparison modes")
    scenarios = {row.get("scenario_id", "").strip() for row in rows}
    missing_scenarios = REQUIRED_SCENARIOS - scenarios
    if missing_scenarios:
        raise ValueError(f"pilot is missing required scenarios: {sorted(missing_scenarios)}")
    for participant in participants:
        participant_rows = [row for row in rows if row["participant_id"].strip() == participant]
        pairs = {(row["scenario_id"].strip(), row["mode"].strip()) for row in participant_rows}
        required_pairs = {(scenario, mode) for scenario in REQUIRED_SCENARIOS for mode in MODES}
        if not required_pairs <= pairs:
            raise ValueError(f"participant {participant} did not evaluate every scenario in both modes")
        orders = {row.get("condition_order", "").strip() for row in participant_rows}
        if not orders <= {"AB", "BA"} or not orders:
            raise ValueError(f"participant {participant} has invalid condition order")

    by_mode: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"].strip()].append(row)
    metrics: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        mode_rows = by_mode[mode]
        correctness = {
            field: _mean(float(_boolean(row[field])) for row in mode_rows)
            for field in BOOLEAN_FIELDS
        }
        extended_correctness = {
            field: _mean(float(_row_boolean(row, field)) for row in mode_rows)
            for field in EXTENDED_BOOLEAN_FIELDS
        }
        all_five = _mean(
            float(all(_boolean(row[field]) for field in BOOLEAN_FIELDS[:5]))
            for row in mode_rows
        )
        times = [float(row["completion_time_sec"]) for row in mode_rows]
        unsupported = [int(row["unsupported_inference_count"]) for row in mode_rows]
        errors = {
            field: _mean(float(_row_boolean(row, field)) for row in mode_rows)
            for field in ERROR_BOOLEAN_FIELDS
        }
        clarity = [float(row.get("subjective_clarity_1_5", 0) or 0) for row in mode_rows]
        cognitive_load = [float(row.get("cognitive_load_1_5", 0) or 0) for row in mode_rows]
        metrics[mode] = {
            "response_count": len(mode_rows),
            "correctness": correctness,
            "extended_correctness": extended_correctness,
            "all_five_correct_rate": all_five,
            "completion_time_sec_mean": _mean(times),
            "completion_time_sec_median": round(statistics.median(times), 6),
            "unsupported_inference_count_total": sum(unsupported),
            "unsupported_inference_rate": _mean(float(value > 0) for value in unsupported),
            "overtrust_rate": errors["overtrust_error"],
            "iou_misinterpretation_rate": errors["iou_misinterpreted_as_probability"],
            "sensitivity_misinterpretation_rate": errors["sensitivity_misinterpreted_as_recommendation"],
            "subjective_clarity_mean": _mean(clarity),
            "cognitive_load_mean": _mean(cognitive_load),
        }

    human = metrics["human_explanation"]
    baseline = metrics["technical_baseline"]
    decision_and_action_all = all(_boolean(row["decision_correct"]) and _boolean(row["action_correct"]) for row in by_mode["human_explanation"])
    correctness_thresholds = all(
        human["correctness"][field] >= 0.8
        for field in ("decision_correct", "reasons_correct", "concern_correct", "action_correct")
    )
    time_ratio = (
        human["completion_time_sec_mean"] / baseline["completion_time_sec_mean"]
        if baseline["completion_time_sec_mean"] > 0
        else float("inf")
    )
    limitations_better = human["correctness"]["concern_correct"] > baseline["correctness"]["concern_correct"]
    overtrust_not_worse = human["overtrust_rate"] <= baseline["overtrust_rate"]
    passed = (
        decision_and_action_all
        and correctness_thresholds
        and human["correctness"]["limitation_correct"] >= 0.8
        and human["unsupported_inference_rate"] == 0.0
        and human["iou_misinterpretation_rate"] == 0.0
        and human["sensitivity_misinterpretation_rate"] == 0.0
        and time_ratio <= 1.25
        and limitations_better
        and overtrust_not_worse
    )
    return {
        "schema_version": "1.0",
        "status": "pass" if passed else "fail",
        "participant_count": len(participants),
        "role_counts": dict(sorted(roles.items())),
        "metrics": metrics,
        "scenario_count": len(scenarios),
        "condition_orders": dict(Counter(row["condition_order"].strip() for row in rows)),
        "claim_allowed": passed,
        "acceptance": {
            "all_participants_identify_decision_and_action": decision_and_action_all,
            "all_five_correct_rate_min": 0.8,
            "limitation_correct_rate_min": 0.8,
            "unsupported_inference_rate_required": 0.0,
            "correctness_thresholds_pass": correctness_thresholds,
            "human_to_baseline_time_ratio": round(time_ratio, 6),
            "time_ratio_max": 1.25,
            "limitations_better_than_baseline": limitations_better,
            "overtrust_not_worse_than_baseline": overtrust_not_worse,
            "iou_misinterpretation_required": 0.0,
            "sensitivity_misinterpretation_required": 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as handle:
        result = score_rows(list(csv.DictReader(handle)))
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
