from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fuzzyxai.final_closure import (
    ConfirmatoryFeatureVector,
    FormativeIteration,
    InvalidActionDecomposition,
    SealedDataset,
    audit_registry,
    compositional_faults,
    conditional_permutation_effect,
    fault_library,
    next_iteration,
    non_refit_ablation,
    refit_ablation,
)
from run_sealed_confirmatory import _parse_vault_payload


D = hashlib.sha256(b"x").hexdigest()
ROOT = Path(__file__).resolve().parents[2]


def _dataset(dataset_id="new", modality="tabular"):
    values = [hashlib.sha256(f"{dataset_id}:{index}".encode()).hexdigest() for index in range(6)]
    return SealedDataset(dataset_id, modality, "source", "license", *values[:5], None, values[-1])


def test_invalid_action_keeps_reason_decomposition() -> None:
    value = InvalidActionDecomposition(route_failure=True, contract_failure=True)
    assert value.operationally_invalid_automatic_action
    assert value.reason_codes == ("route_failure", "contract_failure")


def test_confirmatory_features_require_oof_and_extended_width() -> None:
    ConfirmatoryFeatureVector(D, (0.0,) * 10, (0.0,) * 13, True, "dev-oof")
    with pytest.raises(ValueError, match="out-of-fold"):
        ConfirmatoryFeatureVector(D, (0.0,) * 10, (0.0,) * 13, False, "test")


def test_confirmatory_features_preserve_missing_evidence() -> None:
    value = ConfirmatoryFeatureVector(D, (0.5,) * 8 + (None, None), (None,) * 13, True, "dev-oof")
    assert value.missing_channel_count == 15


def test_dataset_audit_blocks_formative_reuse_label_access_and_overlap() -> None:
    dataset = _dataset()
    report = audit_registry(
        (dataset, _dataset("tab2"), _dataset("img", "image"), _dataset("txt", "text"), _dataset("ts", "timeseries")),
        formative_dataset_ids={"new"}, formative_hashes=set(), oof_object_hashes={D},
        sealed_test_object_hashes={D}, tuning_runner_can_read_test_labels=True,
    )
    assert report["status"] == "blocked"
    assert len(report["blockers"]) == 3


def test_dataset_audit_requires_hashed_nonempty_split_identities() -> None:
    report = audit_registry(
        (_dataset(), _dataset("tab2"), _dataset("img", "image"), _dataset("txt", "text"), _dataset("ts", "timeseries")),
        formative_dataset_ids=set(),
        formative_hashes=set(),
        oof_object_hashes={"raw-object-id"},
        sealed_test_object_hashes=set(),
        tuning_runner_can_read_test_labels=False,
    )
    assert report["status"] == "blocked"
    assert "SEALED_TEST_IDENTITIES_MISSING" in report["blockers"]
    assert "INVALID_OOF_ID_HASH:1" in report["blockers"]


def test_formative_stop_rule_blocks_fourth_iteration() -> None:
    history = tuple(FormativeIteration(index, D, D, D, "reason", ()) for index in (1, 2, 3))
    with pytest.raises(RuntimeError, match="STOP_RULE"):
        next_iteration(history, reason_predeclared=True)


def test_fault_library_has_distinct_templates_and_compositions() -> None:
    assert len(fault_library()) >= 40
    assert len({item.family for item in fault_library()}) == len(fault_library())
    assert len(compositional_faults()) >= 10


def test_rule_ablation_estimands_are_not_mixed() -> None:
    local = non_refit_ablation(0.9, 0.8, support=0.1, redundancy=0.2)
    refit = refit_ablation(0.9, 0.88, support=0.1, redundancy=0.2)
    assert local.estimand != refit.estimand
    effect = conditional_permutation_effect([1, 2, 3, 4], ["a", "a", "b", "b"], lambda x: float(x[::2].sum()), seed=7)
    assert np.isfinite(effect)


def test_final_protocol_freezes_one_zip_endpoint_and_negative_results() -> None:
    protocol = json.loads((ROOT / "study/final_confirmatory_closure/protocol.json").read_text(encoding="utf-8"))
    assert protocol["identifier"] == "FXAI-FINAL-ONE-ZIP-PRACTICAL-CLOSURE"
    assert protocol["primary_review_budget"] == 0.20
    assert protocol["primary_comparator_policy"]
    assert protocol["immutable_results"]["H3-original"] == "not_supported"
    assert protocol["immutable_results"]["H5-P-original"] == "not_supported"
    assert protocol["immutable_results"]["H6-general"] == "not_supported"


def test_ai_scope_does_not_fabricate_review_or_human_validation() -> None:
    scope = json.loads((ROOT / "study/final_confirmatory_closure/ai_text_review_scope.json").read_text(encoding="utf-8"))
    assert scope["status"] == "not_run_not_blocking_technical_release"
    assert scope["review_completed"] is False
    assert scope["review_records"] == 0
    assert scope["ai_review_is_external_validation"] is False
    assert len(scope["disabled_claims"]) == 4


def test_prelock_registry_preserves_unmet_and_unrun_method_boundaries() -> None:
    registry = json.loads((ROOT / "study/final_confirmatory_closure/prelock_method_registry.json").read_text(encoding="utf-8"))
    assert registry["method_status"]["H6-A"] == "formative_target_not_met_method_boundary_preserved"
    assert registry["method_status"]["H6-B"] == "confirmatory_only_requires_two_sealed_tabular_datasets"
    assert registry["method_status"]["H7-B"] == "blocked_projection_stability_not_measured"
    assert registry["confirmatory_claim_allowed"] is False


def test_label_vault_envelope_is_explicitly_unwrapped() -> None:
    assert _parse_vault_payload({"labels": {"object-1": 1}}, "dataset") == {"object-1": "1"}
    with pytest.raises(RuntimeError, match="invalid label-vault envelope"):
        _parse_vault_payload({"object-1": 1}, "dataset")
