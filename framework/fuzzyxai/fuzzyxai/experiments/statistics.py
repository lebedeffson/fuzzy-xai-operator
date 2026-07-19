"""Dependency-light paired statistics for reproducible experiment reports."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from statistics import mean, median, stdev
from typing import Sequence


@dataclass(frozen=True)
class PairedStatistic:
    n_pairs: int
    mean_difference: float
    standard_deviation: float
    median_difference: float
    confidence_interval_95: tuple[float, float]
    wilcoxon_w: float
    wilcoxon_p_two_sided: float
    rank_biserial_effect: float
    worsening_fraction: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    repetitions: int = 4000,
    seed: int = 42,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1)")
    rng = random.Random(seed)
    source = tuple(float(value) for value in values)
    estimates = sorted(mean(rng.choices(source, k=len(source))) for _ in range(repetitions))
    tail = (1.0 - confidence) / 2.0
    lower_index = max(0, int(math.floor(tail * repetitions)))
    upper_index = min(repetitions - 1, int(math.ceil((1.0 - tail) * repetitions)) - 1)
    return estimates[lower_index], estimates[upper_index]


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[indexed[position][0]] = average_rank
        cursor = end
    return ranks


def wilcoxon_signed_rank(differences: Sequence[float]) -> tuple[float, float, float]:
    nonzero = [float(value) for value in differences if abs(float(value)) > 1e-15]
    if not nonzero:
        return 0.0, 1.0, 0.0
    ranks = _average_ranks([abs(value) for value in nonzero])
    positive = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
    negative = sum(rank for rank, value in zip(ranks, nonzero) if value < 0)
    statistic = min(positive, negative)
    total = positive + negative
    effect = (positive - negative) / total if total else 0.0
    n = len(nonzero)
    expected = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    if variance == 0:
        return statistic, 1.0, effect
    correction = 0.5 if statistic < expected else -0.5
    z_score = (statistic - expected + correction) / math.sqrt(variance)
    p_value = math.erfc(abs(z_score) / math.sqrt(2.0))
    return statistic, min(1.0, p_value), effect


def paired_summary(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    higher_is_better: bool = True,
    seed: int = 42,
) -> PairedStatistic:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired samples must be non-empty and equal in length")
    direction = 1.0 if higher_is_better else -1.0
    differences = [direction * (float(new) - float(old)) for old, new in zip(baseline, candidate)]
    statistic, p_value, effect = wilcoxon_signed_rank(differences)
    return PairedStatistic(
        n_pairs=len(differences),
        mean_difference=mean(differences),
        standard_deviation=stdev(differences) if len(differences) > 1 else 0.0,
        median_difference=median(differences),
        confidence_interval_95=bootstrap_mean_ci(differences, seed=seed),
        wilcoxon_w=statistic,
        wilcoxon_p_two_sided=p_value,
        rank_biserial_effect=effect,
        worsening_fraction=sum(value < 0.0 for value in differences) / len(differences),
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original order."""

    count = len(p_values)
    ordered = sorted(enumerate(float(value) for value in p_values), key=lambda item: item[1])
    adjusted = [1.0] * count
    running = 0.0
    for rank, (original_index, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_value))
        adjusted[original_index] = running
    return adjusted


def mcnemar_exact(
    baseline_correct: Sequence[bool],
    candidate_correct: Sequence[bool],
) -> dict[str, float | int]:
    if len(baseline_correct) != len(candidate_correct) or not baseline_correct:
        raise ValueError("paired correctness vectors must be non-empty and equal in length")
    baseline_only = sum(bool(old) and not bool(new) for old, new in zip(baseline_correct, candidate_correct))
    candidate_only = sum(not bool(old) and bool(new) for old, new in zip(baseline_correct, candidate_correct))
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(baseline_only, candidate_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "discordant_pairs": discordant,
        "p_two_sided": p_value,
    }
