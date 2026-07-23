from __future__ import annotations

from decimal import Decimal

from .models import PublicCandidate, R4Gold, R4MethodResult
from .runtime import execute_and_recertify


def score(
    graph: object,
    candidates: tuple[PublicCandidate, ...],
    gold: R4Gold,
    result: R4MethodResult,
) -> dict[str, object]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected = tuple(
        by_id[candidate_id]
        for candidate_id in result.cut
        if candidate_id in by_id
    )
    obligations = {
        obligation
        for candidate in candidates
        for obligation in candidate.covers
    }
    covered = {
        obligation
        for candidate in selected
        for obligation in candidate.covers
    }
    feasible = obligations.issubset(covered)
    cut = tuple(sorted(result.cut))
    membership = (
        gold.status.startswith("CERTIFIED")
        and cut in gold.optimal_cuts
    )
    if gold.optimal_cost is None:
        raw_regret = None
        normalized_regret = 0.0
    elif not feasible:
        raw_regret = None
        normalized_regret = 1.0 + len(obligations - covered) / max(
            1,
            len(obligations),
        )
    else:
        predicted = Decimal(str(result.predicted_cost))
        optimal = Decimal(str(gold.optimal_cost))
        raw_regret = float(predicted - optimal)
        normalized_regret = float(
            (predicted - optimal)
            / max(abs(optimal), Decimal("1e-24"))
        )
    repair = execute_and_recertify(graph, result)
    return {
        "optimal_set_membership": membership,
        "raw_cost_regret": raw_regret,
        "normalized_cost_regret": normalized_regret,
        "obligation_coverage": len(covered) / max(1, len(obligations)),
        "false_certification": (
            result.status == "diagnosed" and not feasible
        ),
        **repair,
    }
