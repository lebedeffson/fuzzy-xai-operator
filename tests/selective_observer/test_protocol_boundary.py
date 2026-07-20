from __future__ import annotations

import json
from pathlib import Path

import pytest

from fuzzyxai.selective_observer import ConfirmatoryProtocolLock


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "study/selective_observer"


def test_frozen_predecessor_and_external_gates_remain_open() -> None:
    cycle = json.loads((STUDY / "research_cycle.json").read_text(encoding="utf-8"))
    gates = json.loads((STUDY / "external_gates.json").read_text(encoding="utf-8"))
    assert cycle["frozen_predecessor"]["commit"] == "e34e52fb8ae62ee1be043d6d5b26a0c9214a0572"
    assert cycle["confirmatory_test_opened"] is False
    assert gates["stable_release_allowed"] is False
    for gate_id in ("domain_language", "comprehension", "expert_action"):
        assert gates[gate_id]["status"] == "open"
        assert gates[gate_id]["raw_records"] == []


def test_hypothesis_templates_cannot_claim_confirmation() -> None:
    for name in ("h3_selective_policy.json", "h5_route_validity.json", "h6_rule_ablation.json"):
        payload = json.loads((STUDY / name).read_text(encoding="utf-8"))
        assert payload["phase"] == "formative_development"
        assert payload["confirmatory_claim_allowed"] is False


def test_protocol_lock_rejects_reused_confirmatory_identity() -> None:
    digest = "a" * 64
    with pytest.raises(ValueError, match="datasets overlap"):
        ConfirmatoryProtocolLock(
            schema_version="1.0",
            frozen_predecessor_commit="1" * 40,
            implementation_commit="2" * 40,
            interface_sha256="b" * 64,
            dictionary_sha256="c" * 64,
            formative_dataset_hashes=(digest,),
            confirmatory_dataset_hashes=(digest,),
            formative_participant_hashes=("d" * 64,),
            confirmatory_participant_hashes=("e" * 64,),
            primary_outcomes=("limitation comprehension",),
            preregistered_baselines=("strong local explainer",),
            minimum_effects=("absolute gain >= 0.15",),
            statistical_tests=("paired preregistered comparison",),
            exclusion_rules=("failed attention check",),
            stopping_rule="fixed sample",
            required_sample_size=40,
            independent_timestamp="2026-08-01T00:00:00Z",
            protocol_sha256="f" * 64,
        )
