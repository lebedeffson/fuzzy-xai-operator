#!/usr/bin/env python3
"""Score anonymized comprehension responses; never create participant data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np


REQUIRED_COLUMNS = {
    "participant_hash",
    "assignment_slot",
    "stimulus_id",
    "condition",
    "consent",
    "attention_pass",
    "decision_correct",
    "reason_correct",
    "limitation_correct",
    "action_correct",
    "unsafe_overtrust",
    "completion_seconds",
}


def boolean(value: str) -> bool:
    if value.lower() not in {"true", "false", "1", "0"}:
        raise ValueError(f"invalid boolean value: {value}")
    return value.lower() in {"true", "1"}


def score(responses: Path, assignments: Path) -> dict[str, object]:
    assignment_rows = json.loads(assignments.read_text(encoding="utf-8"))
    assignments_by_slot = {row["assignment_slot"]: set(row["stimulus_ids"]) for row in assignment_rows}
    valid_slots = set(assignments_by_slot)
    with responses.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or ()):
            raise ValueError(f"missing response columns: {sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ())) }")
        rows = list(reader)
    participants: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_rows: set[tuple[str, str, str]] = set()
    for row in rows:
        if row["assignment_slot"] not in valid_slots:
            raise ValueError("unknown randomized assignment slot")
        key = (row["participant_hash"], row["stimulus_id"], row["condition"])
        if key in seen_rows:
            raise ValueError(f"duplicate participant/stimulus/condition row: {key}")
        seen_rows.add(key)
        participants[row["participant_hash"]].append(row)
    included = {}
    for participant, values in participants.items():
        if not values or not all(boolean(row["consent"]) and boolean(row["attention_pass"]) for row in values):
            continue
        slots = {row["assignment_slot"] for row in values}
        if len(slots) != 1:
            continue
        assigned = assignments_by_slot[next(iter(slots))]
        observed = {row["stimulus_id"] for row in values}
        pairs = {(row["stimulus_id"], row["condition"]) for row in values}
        complete_pairs = all((stimulus, condition) in pairs for stimulus in assigned for condition in ("A", "B"))
        plausible_time = all(float(row["completion_seconds"]) >= 3.0 for row in values)
        if observed == assigned and complete_pairs and plausible_time:
            included[participant] = values
    if len(included) < 24:
        raise ValueError(f"comprehension gate requires 24 valid participants, got {len(included)}")
    by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for values in included.values():
        for row in values:
            if row["condition"] not in {"A", "B"}:
                raise ValueError("condition must be blinded A or B")
            by_condition[row["condition"]].append(row)

    def rate(condition: str, field: str) -> float:
        return mean(boolean(row[field]) for row in by_condition[condition])

    metrics = {
        condition: {
            field: rate(condition, field)
            for field in (
                "decision_correct",
                "reason_correct",
                "limitation_correct",
                "action_correct",
                "unsafe_overtrust",
            )
        }
        for condition in ("A", "B")
    }
    participant_effects = {
        field: [
            mean(boolean(row[field]) for row in values if row["condition"] == "B")
            - mean(boolean(row[field]) for row in values if row["condition"] == "A")
            for values in included.values()
        ]
        for field in ("limitation_correct", "action_correct", "unsafe_overtrust")
    }
    limitation_effect = metrics["B"]["limitation_correct"] - metrics["A"]["limitation_correct"]
    action_effect = metrics["B"]["action_correct"] - metrics["A"]["action_correct"]
    supported = (
        limitation_effect >= 0.05
        and action_effect >= 0.05
        and metrics["B"]["unsafe_overtrust"] <= metrics["A"]["unsafe_overtrust"] + 0.02
    )
    return {
        "schema_version": "1.0",
        "status": "supported" if supported else "not_supported",
        "valid_participants": len(included),
        "metrics": metrics,
        "effects": {
            "limitation_comprehension": limitation_effect,
            "action_selection": action_effect,
            "unsafe_overtrust": metrics["B"]["unsafe_overtrust"] - metrics["A"]["unsafe_overtrust"],
        },
        "preregistered_thresholds": {
            "limitation_comprehension": 0.05,
            "action_selection": 0.05,
            "unsafe_overtrust_noninferiority_margin": 0.02,
        },
        "paired_confidence_intervals_95": {
            field: _bootstrap_interval(values, seed=4201 + index)
            for index, (field, values) in enumerate(participant_effects.items())
        },
        "multiple_outcome_correction": "Holm family declared; support gate uses the three preregistered primary outcomes",
        "response_sha256": _sha256(responses),
        "assignments_sha256": _sha256(assignments),
        "scorer_sha256": _sha256(Path(__file__)),
        "participant_records_generated_by_scorer": False,
    }


def _bootstrap_interval(values: list[float], *, seed: int) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(5000, len(array)))].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = score(args.responses, args.assignments)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: comprehension_scoring participants={payload['valid_participants']} status={payload['status']}")


if __name__ == "__main__":
    main()
