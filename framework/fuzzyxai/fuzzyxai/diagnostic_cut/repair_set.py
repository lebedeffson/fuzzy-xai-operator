"""Repair instructions derived from a diagnostic cut."""

from __future__ import annotations

from dataclasses import dataclass

from .exact_solver import MinimalDiagnosticCut


@dataclass(frozen=True)
class RepairInstruction:
    contract_id: str
    source_path: str
    operation: str


def build_repair_set(cut: MinimalDiagnosticCut) -> tuple[RepairInstruction, ...]:
    source_by_contract = dict(zip(cut.contracts, cut.fault_sources, strict=False))
    return tuple(
        RepairInstruction(contract_id=item, source_path=source_by_contract.get(item, "unknown"), operation=f"restore:{item}")
        for item in cut.contracts
    )
