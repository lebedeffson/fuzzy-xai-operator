from __future__ import annotations

from ..models import Case, GoldRecord


def validate_gold(case: Case, gold: GoldRecord) -> None:
    if gold.case_id != case.case_id:
        raise ValueError("Gold case mismatch")
    if gold.gold_status == "certified":
        if not gold.optimal_cuts or gold.optimal_cost is None:
            raise ValueError("certified Gold lacks optimal solutions")
        obligations = [set(item["candidates"]) for item in case.public_obligations]
        for cut in gold.optimal_cuts:
            if not all(set(cut).intersection(path) for path in obligations):
                raise ValueError("Gold cut does not cover every obligation")
            cost = sum(case.repair_costs[atom] for atom in cut)
            if abs(cost - gold.optimal_cost) > 1e-9:
                raise ValueError("Gold cut cost is inconsistent")

