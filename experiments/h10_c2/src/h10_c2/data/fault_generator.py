from __future__ import annotations

import copy
import random
from typing import Any

from ..models import Transaction


FIELDS = ("version", "schema", "sha256", "available")


def mutate_route(
    clean: dict[str, Any],
    *,
    rng: random.Random,
    mutation_count: int,
    unknown_or_irreparable: bool = False,
) -> tuple[dict[str, Any], tuple[Transaction, ...]]:
    observed = copy.deepcopy(clean)
    transactions: list[Transaction] = []
    targets = rng.sample(observed["nodes"], k=min(mutation_count, len(observed["nodes"])))
    for index, node in enumerate(targets):
        field = rng.choice(FIELDS)
        before = copy.deepcopy(node["observed_attributes"][field])
        after = False if field == "available" else f"mutated-{rng.randrange(10**9):09d}"
        node["observed_attributes"][field] = after
        repairable = not (unknown_or_irreparable and index == 0)
        transactions.append(
            Transaction(
                transaction_id=f"tx:{clean['route_id']}:{index}",
                operation="replace_observed_attribute",
                target_kind="node",
                target_id=node["node_id"],
                field="observed_attributes",
                before=copy.deepcopy({**node["observed_attributes"], field: before}),
                after=copy.deepcopy(node["observed_attributes"]),
                inverse={"field": "observed_attributes", "value": copy.deepcopy({**node["observed_attributes"], field: before})},
                repair_cost=float(1 + (observed["nodes"].index(node) % 5)),
                repairable=repairable,
            )
        )
    return observed, tuple(transactions)


def derive_obligations(
    route: dict[str, Any],
    transactions: tuple[Transaction, ...],
    *,
    equivalent: bool,
) -> tuple[tuple[dict[str, Any], ...], dict[str, float]]:
    nodes = [item["node_id"] for item in route["nodes"]]
    obligations = []
    costs: dict[str, float] = {}
    common = "provider:rebuild-route"
    for transaction in transactions:
        direct = f"{transaction.target_kind}:{transaction.target_id}"
        candidates = [direct]
        costs[direct] = transaction.repair_cost
        if equivalent and len(transactions) > 1:
            candidates.append(common)
            costs[common] = sum(item.repair_cost for item in transactions)
        parent_index = max(0, nodes.index(transaction.target_id) - 1)
        parent = f"node:{nodes[parent_index]}"
        if parent != direct and parent_index % 3 == 0:
            candidates.append(parent)
            costs.setdefault(parent, transaction.repair_cost + 1.0)
        obligations.append(
            {
                "obligation_id": f"obligation:{transaction.transaction_id}",
                "candidates": sorted(set(candidates)),
                "repairable": transaction.repairable,
            }
        )
    return tuple(obligations), costs

