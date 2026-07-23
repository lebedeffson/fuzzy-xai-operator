from __future__ import annotations

from ..hashing import object_sha256
from ..models import Case, GoldRecord
from .equivalent_cuts import enumerate_optimal_cuts
from .transaction_reversal import inverse_action


def derive_gold(case: Case) -> GoldRecord:
    obligations = tuple(tuple(str(atom) for atom in item["candidates"]) for item in case.public_obligations)
    try:
        cuts, cost, _ = enumerate_optimal_cuts(obligations, case.repair_costs)
        status = "certified"
    except (OverflowError, ValueError):
        cuts, cost, status = (), None, "uncertified"
    actions = tuple(inverse_action(transaction) for transaction in case.transactions if transaction.repairable)
    repairable = bool(case.transactions) and all(item.repairable for item in case.transactions)
    payload = {
        "case_id": case.case_id,
        "transactions": [item.__dict__ for item in case.transactions],
        "obligations": case.public_obligations,
        "optimal_cuts": cuts,
        "optimal_cost": cost,
        "allowed_repairs": actions,
    }
    return GoldRecord(
        case_id=case.case_id,
        gold_status=status,
        optimal_cost=cost,
        optimal_cuts=cuts,
        covered_obligations=tuple(str(item["obligation_id"]) for item in case.public_obligations),
        repairable=repairable,
        allowed_repairs=actions,
        oracle_trace_sha256=object_sha256(payload),
    )

