"""Stability metrics for attribution and consensus explanation replicates."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.stats import kendalltau

from .statistics import holm_adjust, paired_bootstrap_difference, paired_permutation_pvalue


def attribution_stability(replicates: Sequence[Sequence[float]], *, top_k: int = 5) -> dict[str, object]:
    matrix = np.asarray(replicates, dtype=float)
    if matrix.ndim != 2 or len(matrix) < 2 or not 1 <= top_k <= matrix.shape[1]:
        raise ValueError("attribution replicates must be a 2D array and top_k must be valid")
    jaccard: list[float] = []
    kendall: list[float] = []
    sign: list[float] = []
    rbo: list[float] = []
    for left_index, right_index in combinations(range(len(matrix)), 2):
        left, right = matrix[left_index], matrix[right_index]
        left_rank = np.argsort(-np.abs(left))
        right_rank = np.argsort(-np.abs(right))
        left_top, right_top = set(left_rank[:top_k]), set(right_rank[:top_k])
        jaccard.append(len(left_top & right_top) / len(left_top | right_top))
        coefficient = kendalltau(np.argsort(left_rank), np.argsort(right_rank)).statistic
        kendall.append(float(0.0 if np.isnan(coefficient) else coefficient))
        shared = sorted(left_top & right_top)
        sign.append(float(np.mean(np.sign(left[shared]) == np.sign(right[shared]))) if shared else 0.0)
        rbo.append(_rank_biased_overlap(left_rank.tolist(), right_rank.tolist(), depth=top_k))
    return {
        "pair_count": len(jaccard),
        "jaccard_at_k": jaccard,
        "kendall_tau": kendall,
        "sign_agreement": sign,
        "rank_biased_overlap": rbo,
        "means": {
            "jaccard_at_k": float(np.mean(jaccard)),
            "kendall_tau": float(np.mean(kendall)),
            "sign_agreement": float(np.mean(sign)),
            "rank_biased_overlap": float(np.mean(rbo)),
        },
    }


def compare_stability(
    baseline_replicates: Sequence[Sequence[float]],
    system_replicates: Sequence[Sequence[float]],
    *,
    baseline_fidelity: Sequence[float],
    system_fidelity: Sequence[float],
    top_k: int = 5,
    practically_null: float = 0.02,
    fidelity_margin: float = -0.02,
    seed: int = 4201,
) -> dict[str, object]:
    baseline = attribution_stability(baseline_replicates, top_k=top_k)
    system = attribution_stability(system_replicates, top_k=top_k)
    endpoints = ("jaccard_at_k", "kendall_tau", "sign_agreement", "rank_biased_overlap")
    raw_p = []
    rows = []
    for offset, endpoint in enumerate(endpoints):
        effect = paired_bootstrap_difference(
            system[endpoint],
            baseline[endpoint],
            repetitions=2000,
            seed=seed + offset,
        )
        p_value = paired_permutation_pvalue(system[endpoint], baseline[endpoint], repetitions=4000, seed=seed + offset)
        raw_p.append(p_value)
        rows.append({"metric": endpoint, **effect, "p_value": p_value})
    adjusted = holm_adjust(raw_p)
    for row, p_value in zip(rows, adjusted, strict=True):
        interval = row["confidence_interval_95"]
        row["holm_adjusted_p"] = p_value
        row["criterion_met"] = bool(row["effect"] > practically_null and interval[0] > 0.0 and p_value < 0.05)
    fidelity = paired_bootstrap_difference(system_fidelity, baseline_fidelity, repetitions=2000, seed=seed + 100)
    fidelity["margin"] = fidelity_margin
    fidelity["noninferior"] = bool(fidelity["confidence_interval_95"][0] >= fidelity_margin)
    return {
        "phase": "formative_only",
        "baseline": baseline["means"],
        "system": system["means"],
        "stability_effects": rows,
        "fidelity_noninferiority": fidelity,
        "formative_target_met": bool(any(row["criterion_met"] for row in rows) and fidelity["noninferior"]),
        "confirmatory_claim_allowed": False,
    }


def _rank_biased_overlap(left: list[int], right: list[int], *, depth: int, persistence: float = 0.9) -> float:
    score = 0.0
    for index in range(1, depth + 1):
        overlap = len(set(left[:index]) & set(right[:index])) / index
        score += (1.0 - persistence) * persistence ** (index - 1) * overlap
    return score / (1.0 - persistence**depth)
