from __future__ import annotations

import json
from pathlib import Path

from experiments.negative_results_remediation.common import FROZEN_NEGATIVE_CLAIMS, verify_protocol


def test_protocol_hash_and_old_negative_claims_are_frozen() -> None:
    assert verify_protocol()
    registry = json.loads(Path("config/negative_remediation_claim_registry.json").read_text())
    assert registry["immutable_claims"] == FROZEN_NEGATIVE_CLAIMS
    assert not registry["manual_positive_override_allowed"]


def test_protocol_forbids_test_tuning_and_target_merging() -> None:
    protocol = json.loads(Path("config/negative_remediation_protocol.json").read_text())
    assert "feature_selection" in protocol["confirmatory_test_forbidden_uses"]
    feature_manifest = json.loads(Path("config/negative_remediation_feature_manifest.json").read_text())
    assert not feature_manifest["model_error_and_structural_targets_may_be_merged"]
    assert not feature_manifest["binary_critical_gap_as_only_h5_feature_allowed"]
