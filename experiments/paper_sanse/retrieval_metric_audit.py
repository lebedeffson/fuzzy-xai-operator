"""Verify the self-retrieval probability calculation in SANSE Table 7.

The query for each of 47 printed entries is made from its own labels.
All candidates sharing those labels are treated as equiprobable.

This is internal ambiguity of a labeled catalog, NOT an estimate of
success for independent users or ground-truth method recommendations.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

CSV = Path(__file__).with_name("figure3_transcription_47.csv")


def metrics(rows: list[dict[str, str]], axes: tuple[str, ...]) -> dict:
    groups = Counter(tuple(row[axis] for axis in axes) for row in rows)
    n = len(rows)
    if n != 47:
        raise ValueError("Expected exactly 47 published printed entries")
    group_count = len(groups)
    collision_pairs = sum(m * (m - 1) // 2 for m in groups.values())
    mean_candidates = sum(m * m for m in groups.values()) / n
    mean_correct_uniform_pick = sum(m * (1/m) for m in groups.values()) / n
    return {
        "axes": axes,
        "n_queries": n,
        "occupied_groups": group_count,
        "collision_pairs": collision_pairs,
        "mean_candidates": mean_candidates,
        "historically_mislabeled_inverse_mean_size": 1 / mean_candidates,
        "correct_mean_uniform_top1_success": mean_correct_uniform_pick,
        "identity_occupied_groups_over_n": group_count / n,
        "extra_candidates_over_queries": 2 * collision_pairs,
    }


def run() -> dict:
    with CSV.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    two = metrics(rows, ("obj", "audience"))
    three = metrics(rows, ("source", "obj", "audience"))
    if (two["occupied_groups"], three["occupied_groups"]) != (14, 24):
        raise AssertionError("Unexpected grouping in Figure 3")
    if (two["collision_pairs"], three["collision_pairs"]) != (80, 45):
        raise AssertionError("Unexpected collision count")
    if (two["extra_candidates_over_queries"], three["extra_candidates_over_queries"]) != (160, 90):
        raise AssertionError("Unexpected extra-candidate count")
    if (round(two["historically_mislabeled_inverse_mean_size"] * 100, 1),
        round(three["historically_mislabeled_inverse_mean_size"] * 100, 1)) != (22.7, 34.3):
        raise AssertionError("Unable to reproduce the previously published quantity")
    if (round(two["correct_mean_uniform_top1_success"] * 100, 1),
        round(three["correct_mean_uniform_top1_success"] * 100, 1)) != (29.8, 51.1):
        raise AssertionError("Average per-query success does not match exact grouping")
    for value in (two, three):
        if abs(value["correct_mean_uniform_top1_success"] -
               value["identity_occupied_groups_over_n"]) > 1e-14:
            raise AssertionError("Mathematical occupied-group identity violated")

    return {
        "data_sha256": hashlib.sha256(CSV.read_bytes()).hexdigest(),
        "two_axes": two,
        "three_axes": three,
        "grouped_names_are_individual_printed_entries": [
            row["method"] for row in rows if "," in row["method"]
        ],
        "limitations": [
            "Original Figure 3 is authored by the study, not independently relabeled",
            "Queries are constructed using the correct answer's own tags",
            "Uniform random selection among returned candidates is an idealized protocol",
            "No inference of real-user effectiveness or independent recommendation correctness",
            "Mention-weighted results for 355 occurrences are outside the displayed data",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
