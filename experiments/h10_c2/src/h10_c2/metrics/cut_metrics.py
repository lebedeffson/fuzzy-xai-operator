from __future__ import annotations

from ..models import GoldRecord, MethodResult


def score_cut(result: MethodResult, gold: GoldRecord, obligations: tuple[dict, ...]) -> dict[str, float | int | bool]:
    predicted = set(result.predicted_cut)
    cuts = [set(cut) for cut in gold.optimal_cuts]
    membership = bool(cuts and predicted in cuts)
    jaccards = [
        len(predicted & cut) / len(predicted | cut) if predicted | cut else 1.0
        for cut in cuts
    ]
    covered = sum(bool(predicted.intersection(item["candidates"])) for item in obligations)
    closest = max(cuts, key=lambda cut: len(predicted & cut) / max(1, len(predicted | cut)), default=set())
    return {
        "optimal_cut_set_membership": membership,
        "cut_cost_regret": (
            result.predicted_cost - gold.optimal_cost if gold.optimal_cost is not None else float("nan")
        ),
        "cut_jaccard_best": max(jaccards, default=float("nan")),
        "obligation_coverage": covered / len(obligations) if obligations else 1.0,
        "extra_elements": len(predicted - closest),
        "missing_elements": len(closest - predicted),
        "optimality_claimed": result.optimality_claimed,
    }

