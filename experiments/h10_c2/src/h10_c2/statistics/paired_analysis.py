from __future__ import annotations

from .clustered_bootstrap import paired_hierarchical_bootstrap
from .multiple_testing import holm_adjust


def analyze_primary(rows: list[dict], baseline: str, repetitions: int, seed: int) -> list[dict]:
    claims = {
        "H10-C2a": "optimal_cut_set_membership",
        "H10-C2b": "full_recertification_success",
    }
    results = []
    raw = {}
    for offset, (claim, metric) in enumerate(claims.items()):
        result = paired_hierarchical_bootstrap(
            rows,
            metric=metric,
            method="fuzzyxai_v21",
            baseline=baseline,
            repetitions=repetitions,
            seed=seed + offset,
        )
        result.update({"claim": claim, "metric": metric, "baseline": baseline})
        raw[claim] = result["p_raw"]
        results.append(result)
    adjusted = holm_adjust(raw)
    for result in results:
        result["p_holm"] = adjusted[result["claim"]]
    return results

