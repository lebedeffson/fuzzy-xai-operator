#!/usr/bin/env python3
"""Fail closed if the formative package claims confirmation or contains fake external records."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "study/selective_observer"
PREDECESSOR = "e34e52fb8ae62ee1be043d6d5b26a0c9214a0572"


def main() -> None:
    cycle = load("research_cycle.json")
    gates = load("external_gates.json")
    manifest = load("protocol_manifest.json")
    if cycle["frozen_predecessor"]["commit"] != PREDECESSOR:
        raise RuntimeError("frozen predecessor identity changed")
    if cycle["current_phase"] != "formative_development" or cycle["confirmatory_test_opened"]:
        raise RuntimeError("formative package cannot open confirmatory data")
    if manifest["confirmatory_protocol_locked"] or manifest["external_records_present"]:
        raise RuntimeError("draft protocol cannot claim a locked or externally completed state")
    for gate_id in ("domain_language", "comprehension", "expert_action"):
        gate = gates[gate_id]
        if gate["status"] != "open" or gate["raw_records"]:
            raise RuntimeError(f"external gate {gate_id} contains unsupported closure evidence")
    if gates["stable_release_allowed"]:
        raise RuntimeError("stable release cannot be enabled during formative development")
    print("PASS: selective_observer_formative_boundary predecessor=frozen external_gates=open stable=false")


def load(name: str) -> dict[str, object]:
    return json.loads((STUDY / name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
