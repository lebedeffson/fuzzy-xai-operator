from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "framework/fuzzyxai"))
from fuzzyxai.doctoral_catalog import AUTHOR_METHODS, catalog_lookup, validate_catalog
from fuzzyxai.doctoral_certification import (
    canonical_json, certify_payload, compare_recertifications, fuzzy_and, project_for_audience,
    select_representation, verify_certificate, _diagnostic_cut, Violation,
)
from fuzzyxai.doctoral_cli import main
from examples.doctoral_certification.make_control_example import build

COORDINATE = {"method_id": "synthetic_rule_control", "source": "intrinsic", "object": "logic",
              "audience": "auditor", "canonical_configuration": "synthetic rule"}
PROFILE = {"profile": ["u_num", "u_ling", "u_int"], "selected_class": "F_int", "measured_delta": 0.05}
POLICY = {"allow_controlled_fixture": True}


def receipt(data=None, *, taxonomy=None, representation=None, policy=None):
    return certify_payload(data if data is not None else build(),
                           taxonomy=taxonomy if taxonomy is not None else COORDINATE,
                           representation=representation if representation is not None else PROFILE,
                           policy=policy if policy is not None else POLICY)


def codes(data):
    return {item["code"] for item in data["decision"]["issues"]}


def test_control_receipt_is_structurally_valid_not_model_accuracy():
    result = receipt()
    assert result["decision"]["status"] == "valid"
    assert result["decision"]["route_certified"] is True
    assert "out-of-sample accuracy" in result["decision"]["not_validated"]
    assert verify_certificate(result)["valid"] is True


def test_fixture_is_not_measured_validation_without_explicit_optin():
    result = receipt(policy={})
    assert result["decision"]["status"] == "insufficient_evidence"
    assert "D_fixture_scope" in codes(result)
    assert not result["decision"]["auto_accept_permitted"]


def test_taxonomy_partition_stable_and_all_three_axes():
    summary = validate_catalog()
    assert summary["count"] == 47
    assert sum(summary["source_counts"].values()) == 47
    assert len({row[0] for row in AUTHOR_METHODS}) == 47


def test_taxonomy_strict_match_for_registered_method():
    reg = catalog_lookup("m27")
    assert reg is not None
    correct = {"method_id": "m27", "source": reg["source"], "object": reg["object"], "audience": reg["audience"], "canonical_configuration": reg["label"]}
    assert receipt(taxonomy=correct)["decision"]["taxonomy"]["catalog_status"] == "registered"
    wrong = {**correct, "source": "intrinsic"}
    assert "D_taxonomy" in codes(receipt(taxonomy=wrong))


def test_taxonomy_missing_axis_is_not_backfilled():
    broken = {**COORDINATE, "object": ""}
    assert receipt(taxonomy=broken)["decision"]["status"] != "valid"


def test_input_provenance_sha_missing_rejected():
    x = build(); x["trace"]["input_sha256"] = ""
    assert "D_trace_hash" in codes(receipt(x))


def test_dataset_version_missing_rejected_by_default():
    x = build(); x["trace"]["dataset_version"] = None
    assert "D_dataset_version" in codes(receipt(x))


def test_timezone_missing_rejected():
    x = build(); x["trace"]["generated_at"] = "2026-11-07T12:00:00"
    assert "D_time" in codes(receipt(x))


def test_duplicate_object_id_rejected():
    x = build(); x["trace"]["object_ids"] = ["sample", "sample"]
    assert "D_object_id" in codes(receipt(x))


def test_dangling_graph_edge_rejected():
    x = build(); x["explanation_graph"]["edges"].append({"source": "unknown", "target": "action"})
    assert "D_dangling_edge" in codes(receipt(x))


def test_provenance_cycle_rejected():
    x = build(); x["explanation_graph"]["edges"].append({"source": "action", "target": "data:synthetic-1"})
    assert "D_graph_cycle" in codes(receipt(x))


def test_supported_claim_requires_real_ref():
    x = build(); x["claims"][0]["evidence_refs"] = ["fabricated-evidence"]
    assert "D_unknown_evidence" in codes(receipt(x))


def test_claim_without_evidence_is_not_supported():
    x = build(); x["claims"][0]["evidence_refs"] = []
    assert "D_claim_ungrounded" in codes(receipt(x))


def test_missing_rule_and_activation_block_full_interface():
    x = build(); x["layers"]["rules"] = []; x["layers"]["fuzzy_rule_activations"] = []
    assert {"D_rule_projection", "D_rule_activation"} <= codes(receipt(x))


def test_missing_linguistic_degrees_never_implies_zero():
    x = build(); x["fuzzy"]["memberships"] = {}
    assert "D_term_projection" in codes(receipt(x))


def test_unknown_membership_origin_rejected():
    x = build(); x["fuzzy"]["membership_policies"]["risk"]["origin"] = "magic"
    assert "D_membership_origin" in codes(receipt(x))


def test_nonfinite_membership_is_not_a_degree():
    x = build(); x["fuzzy"]["memberships"]["synthetic-1"]["high"] = "0.8"
    # Never silently interpret a free-text value as a measured degree.
    assert "D_invalid_numeric" in codes(receipt(x))



def test_actual_tnorm_rechecks_measured_rule_activation():
    x = build(); x["layers"]["fuzzy_rule_activations"][0]["activation_strength"] = 0.72
    assert "D_tnorm_inconsistent" in codes(receipt(x))


def test_missing_rule_antecedents_are_not_hallucinated():
    x = build(); x["layers"]["fuzzy_rule_activations"][0]["terms"] = []
    assert "D_activation_terms_missing" in codes(receipt(x))


def test_contested_claim_blocks_automatic_certification():
    x = build(); x["claims"][0]["evidence_status"] = "contested"
    assert "D_contested_claim" in codes(receipt(x))
    assert not receipt(x)["decision"]["auto_accept_permitted"]


def test_real_graph_requires_witness_to_claim_path():
    x = build(); x["explanation_graph"]["edges"] = [e for e in x["explanation_graph"]["edges"]
        if not (e["source"] == "rule:R1" and e["target"] == "claim:C1")]
    assert "D_unlinked_claim" in codes(receipt(x))


def test_reported_model_sha_must_match_provenance_graph():
    x = build(); x["explanation_graph"]["nodes"][1]["payload"]["model_fingerprint"] = hashlib.sha256(b"other").hexdigest()
    assert "D_model_lineage" in codes(receipt(x))


def test_reported_dataset_version_must_match_provenance_graph():
    x = build(); x["explanation_graph"]["nodes"][0]["payload"]["dataset_version"] = "different"
    assert "D_dataset_lineage" in codes(receipt(x))

def test_alignment_bound_violation():
    x = build(); x["disagreement"]["gamma_max"] = 0.05
    assert "D_alignment" in codes(receipt(x))


def test_alignment_recomputation_catches_tampering():
    x = build(); x["disagreement"]["gamma"] = 0.21
    assert "D_alignment_reconstruction" in codes(receipt(x))


def test_declared_alignment_applicable_missing_measurement_fails_closed():
    x = build(); x["disagreement"]["gamma"] = None; x["disagreement"]["alignment_status"] = "missing"
    assert "D_alignment_not_measured" in codes(receipt(x))


def test_reduction_loss_bound_and_provenance():
    x = build(); x["disagreement"]["delta"] = 0.4
    assert "D_reduction" in codes(receipt(x))
    assert "D_reduction_provenance" in codes(receipt(x))


def test_representation_minimality():
    result = select_representation(["u_num", "u_ling"], audience="auditor")
    assert result["selected_class"] == "F0"
    result = select_representation(["u_num", "u_ling", "u_int", "u_conf"], audience="end_user")
    assert result["selected_class"] == "FML_user"
    result = select_representation(["u_num", "u_ling", "u_conf", "u_trace"], audience="auditor")
    assert result["selected_class"] == "FML_audit"


def test_f0_can_never_cover_interval_plus_source_conflict():
    wrong = {**PROFILE, "profile": ["u_num", "u_ling", "u_int", "u_conf"], "selected_class": "F0"}
    assert "D_choice" in codes(receipt(representation=wrong))


def test_unknown_profile_flag_fails_closed():
    wrong = {**PROFILE, "profile": ["u_num", "impossible_uncertainty"]}
    assert "D_choice" in codes(receipt(representation=wrong))


def test_risk_formula_recomputed():
    x = build(); x["risk"]["rho"] += 0.1
    assert "D_risk_arithmetic" in codes(receipt(x))


def test_risk_weights_not_renormalized():
    x = build(); x["risk"]["risk_weights"]["rho_p"] = 0.50
    assert "D_risk_weight_sum" in codes(receipt(x))


def test_risk_missing_component_never_treated_as_zero():
    x = build(); x["risk"]["components"].pop("u_M")
    assert "D_risk_components" in codes(receipt(x))


def test_risk_nan_is_rejected_not_saved_as_json():
    x = build(); x["risk"]["components"]["u_M"] = float("nan")
    with pytest.raises(ValueError):
        receipt(x)


def test_critical_override_cannot_be_bypassed():
    x = build(); x["risk"]["components"]["chi_R"] = 1
    x["risk"]["chi_r_crit"] = 1
    x["risk"]["rho"] += 0.1
    x["risk"]["action"] = "accept"
    assert "D_critical_not_blocked" in codes(receipt(x))


def test_noncritical_rupture_is_possible_per_chapter_three():
    x = build(); x["risk"]["components"]["chi_R"] = 1
    x["risk"]["chi_r_crit"] = 0
    x["risk"]["rho"] += 0.1
    # Noncritical rupture has a risk penalty but no unconditional block.
    assert "D_critical_mismatch" not in codes(receipt(x))


def test_no_causal_claim_without_causal_evidence():
    x = build()
    tax = {**COORDINATE, "object": "causal"}
    assert "D_causal_claim" in codes(receipt(x, taxonomy=tax))


def test_three_tnorms_and_reject_empty():
    assert fuzzy_and([0.6, 0.8]) == 0.6
    assert fuzzy_and([0.6, 0.8], "product") == pytest.approx(0.48)
    assert fuzzy_and([0.6, 0.8], "lukasiewicz") == pytest.approx(0.4)
    with pytest.raises(ValueError):
        fuzzy_and([])
    with pytest.raises(ValueError):
        fuzzy_and([0.8, float("nan")])


def test_tampering_with_decision_or_sha_detected():
    x = receipt()
    fake = copy.deepcopy(x); fake["decision"]["status"] = "invalid"
    assert not verify_certificate(fake)["valid"]
    fake = copy.deepcopy(x); fake["sha256"] = "0" * 64
    assert not verify_certificate(fake)["valid"]
    fake = copy.deepcopy(x); fake["bindings"]["payload"]["risk"]["action"] = "block"
    assert not verify_certificate(fake)["valid"]


def test_two_replays_are_byte_identical():
    assert canonical_json(receipt()) == canonical_json(receipt())


def test_audience_projections_do_not_recompute_different_actions():
    cert = receipt()
    a = [project_for_audience(cert, role) for role in ("end_user", "developer", "auditor")]
    assert {x["action"] for x in a} == {"accept"}
    assert "trace" not in a[0]
    assert "certificate_sha256" in a[-1]


def test_recertification_requires_independent_verified_inputs():
    broken = build(); broken["risk"]["rho"] += 0.12
    report = compare_recertifications(receipt(broken), receipt())
    assert report["before_status"] == "invalid"
    assert report["after_status"] == "valid"
    assert report["changed_input"] is True
    assert report["recertified"] is True
    assert ["D_risk_arithmetic", "risk.rho"] in report["resolved"]


def test_recertification_does_not_execute_or_fabricate_repair():
    cert = receipt()
    report = compare_recertifications(cert, cert)
    assert report["changed_input"] is False
    assert report["resolved"] == []


def test_exact_cut_finds_joint_cause_not_all_symptoms():
    issues = [
        Violation("A", "a", "", "invalid", ("model", "adapter")),
        Violation("B", "b", "", "invalid", ("model", "trace")),
        Violation("C", "c", "", "invalid", ("model", "trace")),
    ]
    out = _diagnostic_cut(issues, {"model": 1, "adapter": 3, "trace": 3})
    assert out["atoms"] == ["model"]
    assert out["optimality_proven"]
    assert out["cost"] == 1


def test_cli_certify_verify_and_negative(tmp_path):
    examples = ROOT / "examples/doctoral_certification"
    output = tmp_path / "cert"
    rc = main(["certify", "--input", str(examples / "view_model.json"),
               "--taxonomy", str(examples / "taxonomy.json"),
               "--representation", str(examples / "representation.json"),
               "--policy", str(examples / "policy.json"), "--out-dir", str(output)])
    assert rc == 0
    assert (output / "certificate.json").exists()
    assert main(["verify", "--receipt", str(output / "certificate.json")]) == 0
    cert = json.loads((output / "certificate.json").read_text())
    cert["decision"]["action"] = "block"
    (output / "certificate.json").write_text(json.dumps(cert))
    assert main(["verify", "--receipt", str(output / "certificate.json")]) == 3


def test_forged_recommendation_not_certified_as_nonnative_truth():
    x = build(); x["trace"]["result_origin"] = "controlled_fixture"
    assert "D_fixture_scope" in codes(receipt(x, policy={}))
