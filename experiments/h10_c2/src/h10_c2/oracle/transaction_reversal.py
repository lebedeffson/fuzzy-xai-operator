from __future__ import annotations

import copy
from typing import Any

from ..models import Transaction


def apply_transaction(route: dict[str, Any], transaction: Transaction) -> dict[str, Any]:
    result = copy.deepcopy(route)
    collection = "nodes" if transaction.target_kind == "node" else "edges"
    key = "node_id" if transaction.target_kind == "node" else "edge_id"
    target = next(item for item in result[collection] if item[key] == transaction.target_id)
    if transaction.field is None:
        raise ValueError("field is required for benchmark transactions")
    target[transaction.field] = copy.deepcopy(transaction.after)
    return result


def reverse_transaction(route: dict[str, Any], transaction: Transaction) -> dict[str, Any]:
    result = copy.deepcopy(route)
    collection = "nodes" if transaction.target_kind == "node" else "edges"
    key = "node_id" if transaction.target_kind == "node" else "edge_id"
    target = next(item for item in result[collection] if item[key] == transaction.target_id)
    target[str(transaction.inverse["field"])] = copy.deepcopy(transaction.inverse["value"])
    return result


def inverse_action(transaction: Transaction) -> dict[str, Any]:
    return {
        "operation": "restore_from_registered_provider",
        "target_kind": transaction.target_kind,
        "target": transaction.target_id,
        "field": transaction.inverse["field"],
        "provider_ref": f"registry://{transaction.target_id}",
        "cost": transaction.repair_cost,
        "preconditions": [f"provider_available:{transaction.target_id}"],
        "postconditions": [f"contract_revalidated:{transaction.target_id}"],
    }

