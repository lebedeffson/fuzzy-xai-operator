#!/usr/bin/env python3
"""Score blinded expert actions against consensus rather than a single expert."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ACTIONS = ("accept", "review", "block")


def score(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    reviewers = {row["reviewer_hash"] for row in rows}
    objects = {row["object_id"] for row in rows}
    keys = [(row["reviewer_hash"], row["object_id"], row["condition"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate reviewer/object/condition row")
    if len(reviewers) < 3 or len(objects) < 100:
        raise ValueError("expert gate requires at least three reviewers and 100 shared objects")
    for reviewer in reviewers:
        reviewed = {row["object_id"] for row in rows if row["reviewer_hash"] == reviewer}
        if reviewed != objects:
            raise ValueError("every expert must evaluate the same objects")
    human_rows = [row for row in rows if row["condition"] == "expert_only"]
    by_object: dict[str, list[str]] = defaultdict(list)
    for row in human_rows:
        if row["action"] not in ACTIONS:
            raise ValueError("invalid expert action")
        by_object[row["object_id"]].append(row["action"])
    if set(by_object) != objects:
        raise ValueError("expert-only consensus is missing objects")
    consensus = {object_id: Counter(actions).most_common(1)[0][0] for object_id, actions in by_object.items()}
    agreement = _pairwise_agreement(human_rows)
    fleiss_kappa = _fleiss_kappa(by_object)
    methods: dict[str, dict[str, float]] = {}
    for condition in sorted({row["condition"] for row in rows} - {"expert_only"}):
        method_rows = [row for row in rows if row["condition"] == condition]
        if not method_rows:
            continue
        agreements = [row["action"] == consensus[row["object_id"]] for row in method_rows]
        unsafe = [row["action"] == "accept" and consensus[row["object_id"]] != "accept" for row in method_rows]
        false_block = [row["action"] == "block" and consensus[row["object_id"]] != "block" for row in method_rows]
        methods[condition] = {
            "agreement": sum(agreements) / len(agreements),
            "unsafe_accept": sum(unsafe) / len(unsafe),
            "false_block": sum(false_block) / len(false_block),
        }
    adaptive = methods.get("adaptive_fuzzyxai")
    simple = methods.get("strong_simple_baseline")
    if adaptive is None or simple is None:
        raise ValueError("expert gate requires strong_simple_baseline and adaptive_fuzzyxai conditions")
    supported = (
        adaptive["agreement"] - simple["agreement"] >= 0.02
        and adaptive["unsafe_accept"] <= simple["unsafe_accept"] + 0.01
    )
    return {
        "schema_version": "1.0",
        "status": "supported" if supported else "not_supported",
        "reviewer_count": len(reviewers),
        "object_count": len(objects),
        "decision_count": len(rows),
        "consensus": consensus,
        "method_metrics": methods,
        "reliability": {
            "pairwise_agreement": agreement,
            "fleiss_kappa": fleiss_kappa,
            "krippendorff_alpha_nominal": fleiss_kappa,
            "alpha_note": "for a complete balanced nominal panel, nominal alpha equals the same chance-corrected agreement form",
        },
        "preregistered_thresholds": {
            "minimum_agreement_gain": 0.02,
            "unsafe_accept_noninferiority_margin": 0.01,
        },
        "response_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "single_expert_is_gold_standard": False,
        "records_generated_by_scorer": False,
    }


def _pairwise_agreement(rows: list[dict[str, str]]) -> float:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[row["object_id"]].append(row["action"])
    agreements = []
    for actions in grouped.values():
        for left in range(len(actions)):
            agreements.extend(actions[left] == actions[right] for right in range(left + 1, len(actions)))
    return sum(agreements) / len(agreements)


def _fleiss_kappa(grouped: dict[str, list[str]]) -> float:
    counts = np.asarray([[actions.count(action) for action in ACTIONS] for actions in grouped.values()], dtype=float)
    ratings = counts.sum(axis=1)
    if not np.all(ratings == ratings[0]) or ratings[0] < 2:
        raise ValueError("Fleiss kappa requires the same reviewer count for every object")
    observed = np.mean((np.sum(counts**2, axis=1) - ratings) / (ratings * (ratings - 1)))
    proportions = counts.sum(axis=0) / counts.sum()
    expected = float(np.sum(proportions**2))
    return float((observed - expected) / (1.0 - expected)) if expected < 1.0 else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = score(args.responses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: expert_action_scoring reviewers={payload['reviewer_count']} objects={payload['object_count']}")


if __name__ == "__main__":
    main()
