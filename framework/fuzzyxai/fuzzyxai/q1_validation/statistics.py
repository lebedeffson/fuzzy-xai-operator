"""Pre-registered statistics, including fidelity non-inferiority."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from fuzzyxai.experiments.statistics import bootstrap_mean_ci, wilcoxon_signed_rank

from .schemas import FidelityPair


@dataclass(frozen=True)
class NonInferiorityResult:
    n_pairs: int
    margin: float
    mean_difference: float
    confidence_interval_95: tuple[float, float]
    lower_bound: float
    noninferior: bool
    worsening_beyond_margin_fraction: float
    wilcoxon_w: float
    wilcoxon_p_two_sided: float
    rank_biserial_effect: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def noninferiority_test(
    pairs: Sequence[FidelityPair],
    *,
    margin: float = -0.02,
    seed: int = 4201,
    bootstrap_repetitions: int = 4000,
) -> NonInferiorityResult:
    if not pairs:
        raise ValueError("non-inferiority requires paired observations")
    if margin >= 0.0:
        raise ValueError("non-inferiority margin must be negative")
    differences = [item.difference for item in pairs]
    interval = bootstrap_mean_ci(differences, repetitions=bootstrap_repetitions, seed=seed)
    statistic, p_value, effect = wilcoxon_signed_rank(differences)
    mean_difference = sum(differences) / len(differences)
    return NonInferiorityResult(
        n_pairs=len(pairs),
        margin=margin,
        mean_difference=mean_difference,
        confidence_interval_95=interval,
        lower_bound=interval[0],
        noninferior=interval[0] >= margin,
        worsening_beyond_margin_fraction=sum(value < margin for value in differences) / len(differences),
        wilcoxon_w=statistic,
        wilcoxon_p_two_sided=p_value,
        rank_biserial_effect=effect,
    )
