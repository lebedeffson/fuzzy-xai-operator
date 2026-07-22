from __future__ import annotations

from .mutation_transaction import MutationTransaction


def derive_source_truth(transactions: tuple[MutationTransaction, ...]) -> tuple[str, ...]:
    """Return only elements directly changed by executed transactions."""
    return tuple(
        dict.fromkeys(
            element
            for transaction in transactions
            for element in transaction.changed_nodes + transaction.changed_edges
        )
    )
