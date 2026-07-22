from __future__ import annotations

import json
from typing import Any

from .mutation_transaction import MutationTransaction


def canonical_repair_action(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def derive_repair_truth(transactions: tuple[MutationTransaction, ...]) -> tuple[str, ...]:
    """Return inverse transactions in reverse execution order."""
    actions: list[dict[str, Any]] = []
    for transaction in reversed(transactions):
        inverse = transaction.inverse_operation
        if inverse["operation"] == "restore_edges":
            actions.extend({"operation": "remove_edge", "source": source, "target": target} for source, target in inverse["remove"])
            actions.extend({"operation": "add_edge", "source": source, "target": target} for source, target in inverse["add"])
        elif inverse["operation"] == "restore_node":
            actions.append({"operation": "restore_node", "node": inverse["node"]})
            actions.extend({"operation": "add_edge", "source": edge["source"], "target": edge["target"]} for edge in inverse["edges"])
        else:
            actions.append(inverse)
    return tuple(canonical_repair_action(item) for item in actions)
