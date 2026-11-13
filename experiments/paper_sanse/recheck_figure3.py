"""Recompute unweighted SANSE method statistics from published Figure 3.

47 rows were transcribed from the article's displayed S/O/A grid, one
printed entry per configuration. This is NOT the original literature
database: the 355 mention weights and independent annotation are absent.
No imputation of unavailable source observations is performed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = ("internal", "embedded", "posthoc")
O = ("features", "representations", "data", "logic", "causal")
A = ("user", "developer", "researcher", "auditor", "regulator")
CELL_COUNTS = ((3, 0, 1, 4, 1), (1, 3, 2, 4, 1), (13, 4, 4, 5, 1))


def read() -> list[dict[str, str]]:
    with (HERE / "figure3_transcription_47.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 47 or len(set(r["method"] for r in rows)) != 47:
        raise ValueError("Figure 3 requires 47 distinct printed entries")
    if any(r["source"] not in S or r["obj"] not in O or r["audience"] not in A for r in rows):
        raise ValueError("Out-of-vocabulary figure code")
    counts = tuple(tuple(sum(r["source"] == s and r["obj"] == o for r in rows) for o in O) for s in S)
    if counts != CELL_COUNTS:
        raise ValueError(f"Figure 3 cell-count mismatch: {counts}")
    return rows


def entropy(values) -> float:
    counts = Counter(values)
    n = sum(counts.values())
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def mi(rows, x, y) -> float:
    return entropy(r[x] for r in rows) + entropy(r[y] for r in rows) - entropy((r[x], r[y]) for r in rows)


def collisions(rows, *axes) -> int:
    c = Counter(tuple(r[a] for a in axes) for r in rows)
    return sum(v * (v - 1) // 2 for v in c.values())


def shuffle(rows, key, rng):
    values = [r[key] for r in rows]
    rng.shuffle(values)
    return [{**r, key: value} for r, value in zip(rows, values)]


def gain(rows):
    pairs = min(collisions(rows, *axes) for axes in (("source", "obj"), ("source", "audience"), ("obj", "audience")))
    return pairs - collisions(rows, "source", "obj", "audience")


def negative_control(rows, n=50000, seed=20261130):
    rng = random.Random(seed)
    observed = gain(rows)
    extremes = total = 0
    for _ in range(n):
        value = gain(shuffle(rows, "audience", rng))
        extremes += value >= observed
        total += value
    return {"observed_gain": observed, "null_mean": total / n,
            "p_plus_one": (extremes + 1) / (n + 1),
            "permutations": n, "seed": seed, "extremes": extremes}


def loo_audience(rows, tie):
    correct = 0
    for i, row in enumerate(rows):
        train = rows[:i] + rows[i+1:]
        glob = Counter(r["audience"] for r in train)
        local = Counter(r["audience"] for r in train if (r["source"], r["obj"]) == (row["source"], row["obj"]))
        if not local:
            guess = sorted(glob, key=lambda a: (-glob[a], a))[0]
        else:
            m = max(local.values())
            winners = sorted(a for a in local if local[a] == m)
            if tie == "abstain" and len(winners) > 1:
                guess = "UNKNOWN"
            elif tie == "global":
                guess = sorted(winners, key=lambda a: (-glob[a], a))[0]
            else:
                guess = winners[0]
        correct += guess == row["audience"]
    return correct


def run():
    rows = read()
    hist = {axis: entropy(r[axis] for r in rows) for axis in ("source", "obj", "audience")}
    pair = {"+".join(keys): collisions(rows, *keys) for keys in
            (("source",), ("obj",), ("audience",), ("source", "obj"),
             ("source", "audience"), ("obj", "audience"),
             ("source", "obj", "audience"))}
    scores = {x+"_"+y: mi(rows, x, y) for x, y in
              (("source", "obj"), ("source", "audience"), ("obj", "audience"))}
    archive_hist = {"source":1.406, "obj":2.115, "audience":2.069}
    archive_pairs = {"source":442, "obj":259, "audience":269, "source+obj":119,
                     "obj+audience":80, "source+obj+audience":45}
    archive_mi = {"source_obj":0.1633, "source_audience":0.3771, "obj_audience":0.6082}
    assert all(abs(hist[k]-v)<0.0005 for k,v in archive_hist.items())
    assert all(pair[k]==v for k,v in archive_pairs.items())
    assert all(abs(scores[k]-v)<0.00005 for k,v in archive_mi.items())
    joint_h = {
        "source_obj": entropy((r["source"],r["obj"]) for r in rows),
        "all_three": entropy((r["source"],r["obj"],r["audience"]) for r in rows)
    }
    assert abs(joint_h["source_obj"]-3.358)<.0005
    assert abs(joint_h["all_three"]-4.291)<.0005
    loo_mi = {x+"_"+y: [
        min(mi(rows[:i]+rows[i+1:], x, y) for i in range(47)),
        max(mi(rows[:i]+rows[i+1:], x, y) for i in range(47))
    ] for x,y in (("source","obj"),("source","audience"),("obj","audience"))}
    assert abs(loo_mi["source_obj"][0]-.1503)<.00005
    assert abs(loo_mi["source_obj"][1]-.2395)<.00005
    assert abs(loo_mi["source_audience"][0]-.3471)<.00005
    assert abs(loo_mi["source_audience"][1]-.4531)<.00005
    assert abs(loo_mi["obj_audience"][0]-.5544)<.00005
    assert abs(loo_mi["obj_audience"][1]-.6650)<.00005
    ties = {rule: loo_audience(rows, rule) for rule in ("abstain", "alphabetical", "global")}
    assert ties=={"abstain":15, "alphabetical":17, "global":23}
    null = negative_control(rows)
    assert null["observed_gain"] == 35 and .49 < null["p_plus_one"] < .55
    return {
        "status": "UNWEIGHTED_STATISTICS_RECONSTRUCTED_FROM_PUBLISHED_FIGURE",
        "n_entries": 47, "entropy_bits": hist, "collision_pairs": pair,
        "mutual_information_bits": scores, "joint_entropy_bits": joint_h,
        "loo_mi_range_bits": loo_mi,
        "average_candidates_two_axes": 1 + 2 * pair["obj+audience"] / 47,
        "average_candidates_three_axes": 1 + 2 * pair["source+obj+audience"] / 47,
        "negative_control_50000": null,
        "loo_audience_tie_rule_sensitivity": ties,
        "source_image_sha256": "7015f76b602fc13646e2dc6e35bc08cb0127beee274c9a3973d569462fb63106",
        "input_csv_sha256": hashlib.sha256((HERE/"figure3_transcription_47.csv").read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "limits": [
            "Author-produced published figure is NOT the original independently validated data",
            "One figure line is treated as one configuration even if it names grouped methods",
            "355 mentions, bibliographic source rows and weights are unavailable",
            "Original random seed and LOO tie-breaking policy are unknown",
            "No independent evidence for external user benefit, ground-truth labels or original source selection"
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, sort_keys=True))
