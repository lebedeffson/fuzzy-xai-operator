"""Chapter 3: do not confuse interval type-1 and interval type-2 FOU."""
from __future__ import annotations

import copy

import pytest

from fuzzyxai import FuzzyXAI
from fuzzyxai.doctoral_certification import (
    CLASS_COVERAGE, PROFILE_FLAGS, certify_payload, select_representation,
    verify_certificate,
)
from fuzzyxai.interval_type2 import content_sha256, make_it2_receipt
from tests.test_it2_runtime_adapter import model


TAXONOMY = {
    "method_id": "synthetic_rule_control",
    "source": "intrinsic",
    "object": "logic",
    "audience": "auditor",
    "canonical_configuration": "synthetic rule",
}
PROFILE = ["u_num", "u_ling", "u_int", "u_fou"]


def setup_certified_input():
    native = model()
    obj_id, version = "audited-record", "controlled-it2-protocol-v1"
    result = FuzzyXAI.wrap(native).explain_one(
        [0.4], object_id=obj_id, dataset_version=version)
    digest = content_sha256(native.to_dict())
    receipt = make_it2_receipt(
        native, [0.4], source_id=obj_id, dataset_version=version,
        expected_model_sha256=digest)
    representation = {
        "profile": PROFILE,
        "selected_class": "F_IT2",
        "it2_trusted_model_sha256": digest,
        "it2_receipts": [receipt],
    }
    return result, representation


def codes(cert):
    return {issue["code"] for issue in cert["decision"]["issues"]}


def run_cert(result, representation):
    # A Type-2 model can have other missing chapter 2/4 evidence. The test
    # verifies the TYPE-2 portion, not a global/clinical success label.
    return certify_payload(result.to_dict(), taxonomy=TAXONOMY,
                           representation=representation,
                           policy={"allow_controlled_fixture": True})


def test_type2_uncertainty_is_not_type1_interval():
    assert "u_fou" in PROFILE_FLAGS
    assert "u_fou" not in CLASS_COVERAGE["F_int"]
    assert "u_fou" not in CLASS_COVERAGE["FML_audit"]
    selection = select_representation(PROFILE, audience="auditor")
    assert selection["selected_class"] == "F_IT2"
    assert select_representation(["u_num", "u_ling", "u_int"], audience="auditor")["selected_class"] == "F_int"
    assert select_representation(PROFILE + ["u_conf"], audience="auditor")["status"] == "D_choice"


def test_native_type2_inference_is_present_in_route_not_just_configuration():
    result, representation = setup_certified_input()
    assert result.view_model.fuzzy["selected_representation"] == "F_IT2"
    states = [item for item in result.view_model.route
              if item.get("id") == "it2_representation"]
    assert len(states) == 1
    assert states[0]["value_source"] == "computed"
    assert states[0]["output_values"]["selected_class"] == "F_IT2"
    output = run_cert(result, representation)
    replay = output["decision"]["representation"]["it2_verification"]
    assert replay["status"] == "replayed"
    assert replay["replayed_cases"] == 1
    assert not any(code.startswith("D_it2_") for code in codes(output))
    assert not any(code.startswith("D_representation_") for code in codes(output))
    assert verify_certificate(output)["valid"] is True
    # The general doctorate certification could still have unrelated gaps,
    # because a verified type-2 receipt is not a clinical certificate.
    assert output["decision"]["scope"] == (
        "computational_route_compatibility_not_prediction_accuracy_or_causality"
    )


def test_claimed_type2_without_real_replay_is_not_certified():
    result, representation = setup_certified_input()
    incomplete = dict(representation)
    incomplete.pop("it2_receipts")
    decision = run_cert(result, incomplete)
    assert "D_it2_receipts_missing" in codes(decision)
    assert decision["decision"]["route_certified"] is False
    assert decision["decision"]["representation"]["it2_verification"]["status"] == "insufficient_evidence"


def test_type1_label_cannot_hide_recorded_type2_fou():
    result, representation = setup_certified_input()
    fake = dict(representation)
    fake["profile"] = ["u_num", "u_ling", "u_int"]
    fake["selected_class"] = "F_int"
    output = run_cert(result, fake)
    assert "D_it2_profile_omits_fou" in codes(output)
    assert output["decision"]["route_certified"] is False


def test_tampering_with_receipt_changes_certification_status():
    result, representation = setup_certified_input()
    tampered = copy.deepcopy(representation)
    tampered["it2_receipts"][0]["inference"]["probability_lower"] = 0.999
    output = run_cert(result, tampered)
    assert "D_it2_replay_failed" in codes(output)
    assert not output["decision"]["auto_accept_permitted"]


def test_untrusted_or_mismatching_model_anchor_fails_closed():
    result, representation = setup_certified_input()
    mismatched = dict(representation)
    mismatched["it2_trusted_model_sha256"] = "0" * 64
    output = run_cert(result, mismatched)
    assert "D_it2_anchor_mismatch" in codes(output)
    assert "D_it2_replay_failed" in codes(output)
    missing = dict(representation)
    missing.pop("it2_trusted_model_sha256")
    output = run_cert(result, missing)
    assert "D_it2_anchor_missing" in codes(output)


def test_document_model_does_not_justify_execution_by_itself():
    result, representation = setup_certified_input()
    altered = copy.deepcopy(result.to_dict())
    altered["route"] = [item for item in altered["route"]
                        if item.get("id") != "it2_representation"]
    output = run_cert_payload(altered, representation)
    assert "D_representation_not_executed" in codes(output)


def run_cert_payload(payload, representation):
    return certify_payload(payload, taxonomy=TAXONOMY,
                           representation=representation,
                           policy={"allow_controlled_fixture": True})
