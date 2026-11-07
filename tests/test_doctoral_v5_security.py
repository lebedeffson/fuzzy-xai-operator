"""Adversarial replay and cross-operator integrity tests (chapter 1-4, v5).

All inputs are artificial fixtures: these tests prove the checker rejects
inconsistent evidence, not that any classifier or clinical model is correct.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from examples.doctoral_certification.make_control_example import build
from fuzzyxai.doctoral_certification import certify_payload, verify_certificate

TAXONOMY = {"method_id": "synthetic_rule_control", "source": "intrinsic", "object": "logic",
            "audience": "auditor", "canonical_configuration": "synthetic rule"}
REPRESENTATION = {"profile": ["u_num", "u_ling", "u_int"], "selected_class": "F_int", "measured_delta": 0.05}
POLICY = {"allow_controlled_fixture": True}


def cert(payload, *, representation=None, policy=None):
    return certify_payload(payload, taxonomy=TAXONOMY, representation=representation or REPRESENTATION,
                           policy=POLICY if policy is None else policy)


def issues(certified):
    return {item["code"] for item in certified["decision"]["issues"]}


def test_unmodified_control_is_certified_with_explicit_fixture_scope():
    certificate = cert(build())
    assert certificate["decision"]["status"] == "valid"
    assert verify_certificate(certificate)["valid"]
    assert certificate["bindings"]["payload"]["trace"]["result_origin"] == "controlled_fixture"


def test_missing_claim_materialization_cannot_be_hidden_with_direct_action_edge():
    x = build()
    graph = x["explanation_graph"]
    graph["nodes"] = [n for n in graph["nodes"] if n["node_id"] != "claim:C1"]
    graph["edges"] = [e for e in graph["edges"] if "claim:C1" not in (e["source"], e["target"])]
    graph["edges"].append({"source": "prediction", "target": "action"})
    assert "D_claim_node_missing" in issues(cert(x))


def test_activation_cannot_claim_a_rule_that_never_existed():
    x = build()
    x["layers"]["fuzzy_rule_activations"][0]["rule_id"] = "NONEXISTENT"
    assert "D_activation_unregistered_rule" in issues(cert(x))


def test_rule_cannot_be_absent_from_provenance_graph():
    x = build()
    x["layers"]["rules"][0]["rule_id"] = "NONEXISTENT"
    assert "D_rule_node_missing" in issues(cert(x))


def test_activation_object_must_be_in_the_same_input_trace():
    x = build()
    x["layers"]["fuzzy_rule_activations"][0]["object_id"] = "another-patient"
    assert "D_activation_unknown_object" in issues(cert(x))


def test_membership_projection_must_equal_measured_antecedent():
    x = build()
    x["fuzzy"]["memberships"]["synthetic-1"]["high"] = 0.1
    assert "D_membership_projection_mismatch" in issues(cert(x))


def test_self_declared_fuzzy_class_without_route_is_not_execution():
    x = build()
    del x["route"]
    assert "D_representation_not_executed" in issues(cert(x))


def test_operator_class_mismatch_cannot_be_hid_in_view_model():
    x = build()
    x["route"][0]["output_values"]["selected_class"] = "F0"
    assert "D_representation_mismatch" in issues(cert(x))


def test_manual_route_output_not_computed_evidence():
    x = build()
    x["route"][0]["value_source"] = "user_supplied"
    assert "D_representation_untrusted_source" in issues(cert(x))


def test_disabling_any_certification_gate_prevents_full_compliance():
    for relaxation in ("require_risk", "require_graph", "require_rule_interface", "require_representation_execution"):
        disabled = {**POLICY, relaxation: False}
        result = cert(build(), policy=disabled)
        assert result["decision"]["status"] != "valid", relaxation
        assert not result["decision"]["auto_accept_permitted"]
        assert "D_policy_relaxed" in issues(result)


def test_invalid_declared_delta_becomes_a_diagnostic_not_an_exception():
    representation = {**REPRESENTATION, "measured_delta": "undefined"}
    result = cert(build(), representation=representation)
    assert result["decision"]["status"] == "invalid"
    assert "D_choice" in issues(result)
    assert "D_reduction_provenance_invalid" in issues(result)


def test_incompatible_graph_input_prediction_lineage_is_rejected():
    x = build()
    x["explanation_graph"]["edges"] = [edge for edge in x["explanation_graph"]["edges"]
                                        if not (edge["source"] == "data:synthetic-1" and edge["target"] == "prediction")]
    assert "D_input_prediction_lineage" in issues(cert(x))


def test_external_evidence_reference_must_reach_claim_node():
    x = build()
    x["claims"][0]["evidence_refs"] = ["source:R1"]
    good = cert(x)
    assert good["decision"]["status"] == "valid"
    x["explanation_graph"]["edges"] = [e for e in x["explanation_graph"]["edges"]
                                           if not (e["source"] == "rule:R1" and e["target"] == "claim:C1")]
    # Keep prediction-to-action connected: the failure must be the evidence witness.
    x["explanation_graph"]["edges"].append({"source": "prediction", "target": "action"})
    assert "D_unlinked_claim" in issues(cert(x))


def test_certificate_receipt_is_invalid_if_edited_after_issue():
    receipt = cert(build())
    tampered = deepcopy(receipt)
    tampered["decision"]["route_certified"] = False
    assert not verify_certificate(tampered)["valid"]


def test_original_fixture_is_never_accepted_as_measured_without_opt_in():
    result = cert(build(), policy={})
    assert result["decision"]["status"] == "insufficient_evidence"
    assert "D_fixture_scope" in issues(result)
