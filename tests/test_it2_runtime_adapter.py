"""Real FuzzyXAI.wrap -> IT2 inference -> evidence graph -> guarded action."""
from __future__ import annotations

from dataclasses import replace

import pytest

from fuzzyxai import FuzzyXAI, IntervalType2Adapter
from fuzzyxai.interval_type2 import (
    IT2FeatureRule, IT2FuzzyLogit, IT2Shoulder, content_sha256,
    verify_it2_receipt,
)


def model(*, high_coefficient: float = 4.0) -> IT2FuzzyLogit:
    return IT2FuzzyLogit(
        intercept=0.0,
        n_input_features=1,
        features=(
            IT2FeatureRule(
                feature_index=0,
                feature_name="observed_biomarker",
                low_coefficient=-4.0,
                high_coefficient=high_coefficient,
                membership=IT2Shoulder(
                    left_min=0.0, left_max=0.2,
                    right_min=0.6, right_max=0.8,
                    nominal_left=0.1, nominal_right=0.7,
                ),
            ),
        ),
        source_fit="controlled_synthetic_fit_no_clinical_claim",
    )


def test_auto_wrap_discovers_it2_and_predicts_using_type2_point_estimate():
    fuzzy = FuzzyXAI.wrap(model())
    assert isinstance(fuzzy.model_adapter, IntervalType2Adapter)
    adapter = fuzzy.model_adapter
    assert adapter.adapter_id == "interval_type2_v2"
    assert not adapter.output_schema().calibrated_probability
    assert adapter.capabilities().native_rules
    assert adapter.capabilities().uncertainty
    assert not adapter.capabilities().calibration
    pred = adapter.predict([[0.4]])
    inference = model().infer([0.4])
    assert pred.probabilities[0][1] == pytest.approx(inference["probability_midpoint"])
    assert pred.metadata["calibrated_probability"] is False
    assert adapter.model_fingerprint() == content_sha256(model().to_dict())
    assert adapter.model_fingerprint() != IntervalType2Adapter(model(high_coefficient=3.0)).model_fingerprint()


def test_interval_policy_actually_restrains_final_public_action_and_graph():
    fuzzy = FuzzyXAI.wrap(model())
    result = fuzzy.explain_one([0.4], object_id="caseA",
                               dataset_version="synthetic-wdbc-fixture-v1",
                               include_similar_cases=False)
    evidence = result.prediction.metadata["it2_guard"]
    assert evidence["status"] == "verified"
    assert evidence["scope"] == "parameter_FOU_not_predictive_coverage"
    assert evidence["action"] == "request_more_data"
    assert result.action != "accept"
    assert result.view_model.risk["it2_guard"] == evidence
    assert evidence["cases"][0]["replay_verified"] is True
    assert len(evidence["cases"][0]["receipt_sha256"]) == 64
    assert "input_values" not in str(evidence)
    graph = result.explanation_graph
    assert any(node.node_id == "it2:guard" for node in graph.nodes)
    assert any(edge.source == "it2:guard" and edge.target == "action" for edge in graph.edges)
    assert "interval_type2_v2" in result.adapter_id


def test_it2_without_explicit_dataset_version_fails_closed():
    result = FuzzyXAI.wrap(model()).explain_one(
        [0.4], object_id="missing_dataset", include_similar_cases=False)
    assert result.action == "block"
    assert result.prediction.metadata["it2_guard"]["status"] == "insufficient_provenance"
    assert "it2_dataset_provenance" in result.required_missing_channels or any(
        item.get("code") == "D_it2_source_missing" for item in result.view_model.diagnostics
    )


def test_fuzzy_memberships_are_native_evidence_not_surrogate_attributions():
    adapter = IntervalType2Adapter(model())
    pred = adapter.predict([[0.4]])
    evidence = adapter.extract_local_evidence([[0.4]], pred, context=None)
    payload = evidence.to_runtime_mapping()
    assert len(payload["activated_rules"]) == 2
    assert payload["activated_rules"][0]["terms"][0]["feature_value"] == pytest.approx(0.4)
    assert payload["it2_model_sha256"] == adapter.model_fingerprint()
    assert "contributions" not in payload
    assert "it2_feature_evidence" in payload


def test_explicit_adapter_alias_and_batch_guard():
    fuzzy = FuzzyXAI.wrap(model(), adapter="it2")
    res = fuzzy.explain([[0.0], [0.4]], object_ids=["left", "middle"],
                        dataset_version="controlled-v1")
    guard = res.prediction.metadata["it2_guard"]
    assert len(guard["cases"]) == 2
    assert guard["cases"][0]["decision"]["action"] == "accept"
    assert guard["cases"][1]["decision"]["action"] == "request_more_data"
    assert res.action != "accept"


def test_invalid_observations_rejected_before_audit():
    fuzzy = FuzzyXAI.wrap(model())
    with pytest.raises(ValueError):
        fuzzy.explain_one([float("nan")], dataset_version="controlled-v1")
    with pytest.raises(ValueError):
        fuzzy.explain_one([0.1, 0.5], dataset_version="controlled-v1")


def test_user_explanation_is_a_projection_of_verified_it2_evidence():
    result = FuzzyXAI.wrap(model()).explain_one(
        [0.4], object_id="caseA", dataset_version="controlled-v1")
    user = result.explain_it2(audience="domain_user")
    auditor = result.explain_it2(audience="auditor")
    assert user is not None and auditor is not None
    assert user["intervals"] == auditor["intervals"]
    assert "не" in user["limitation"]
    assert "нечёткостью параметров" in user["message"]
    assert user["it2_action"] == "request_more_data"
    assert user["final_action"] == result.action
    compact = result.to_dict(detail="compact")
    standard = result.to_dict(detail="standard")
    assert compact["it2_explanation"] == user
    assert standard["it2_explanation"] == user
    assert "input_values" not in repr(compact["it2_explanation"])


def test_unversioned_provenance_has_explicit_human_warning():
    result = FuzzyXAI.wrap(model()).explain_one([0.4], object_id="caseB")
    text = result.explain_it2()
    assert text is not None
    assert text["status"] == "insufficient_provenance"
    assert "запрещено" in text["message"]
    assert text["final_action"] == "block"

def test_it2_actual_uncertainty_width_enters_declared_risk_observer():
    fuzzy = FuzzyXAI.wrap(model())
    assert fuzzy.plan.uncertainty_policy.method == "it2_parameter_width"
    assert "uncalibrated" in fuzzy.plan.uncertainty_policy.source
    result = fuzzy.explain_one([0.4], object_id="width001",
                               dataset_version="synthetic-fixture-v1")
    observed = result.view_model.risk["components"].get("uncertainty")
    expected = model().infer([0.4])["induced_width"]
    assert observed is not None
    assert observed == pytest.approx(expected)
    assert result.view_model.risk["uncertainty_policy"]["method"] == "it2_parameter_width"


def test_explicit_uncertainty_policy_is_not_overridden_by_auto_it2():
    from fuzzyxai.core.explain_plan import ExplainPlan
    manual = ExplainPlan.default()
    assert manual.uncertainty_policy.method == "none"
    fuzzy = FuzzyXAI.wrap(model(), explain_plan=manual)
    assert fuzzy.plan is manual
    assert fuzzy.plan.uncertainty_policy.method == "none"

def test_external_model_hash_anchor_enforces_model_identity():
    from fuzzyxai.core.explain_plan import ExplainPlan
    true_model = model()
    plan = ExplainPlan.default()
    plan.metadata["it2_expected_model_sha256"] = content_sha256(true_model.to_dict())
    approved = FuzzyXAI.wrap(true_model, explain_plan=plan).explain_one(
        [0.4], object_id="identity-1", dataset_version="fixture-v1")
    guard = approved.prediction.metadata["it2_guard"]
    assert guard["model_anchor_status"] == "matched_declared_external_SHA256"
    assert all(item["model_anchor_status"] == "matched_declared_external_SHA256" for item in guard["cases"])
    assert approved.explain_it2()["model_anchor_status"] == "matched_declared_external_SHA256"

    mismatched = FuzzyXAI.wrap(model(high_coefficient=3.0), explain_plan=plan)
    with pytest.raises(ValueError, match="external anchor"):
        mismatched.explain_one([0.4], object_id="identity-1", dataset_version="fixture-v1")


def test_numeric_replay_does_not_pretend_external_authentication():
    result = FuzzyXAI.wrap(model()).explain_one(
        [0.4], object_id="no-external-anchor", dataset_version="controlled-v1")
    guard = result.prediction.metadata["it2_guard"]
    assert guard["model_anchor_status"] == "in_process_model_only_no_external_authentication"
    assert guard["cases"][0]["replay_verified"] is True


def test_malformed_external_fingerprint_is_rejected():
    from fuzzyxai.core.explain_plan import ExplainPlan
    plan = ExplainPlan.default()
    plan.metadata["it2_expected_model_sha256"] = "untrusted"
    with pytest.raises(ValueError, match="SHA256"):
        FuzzyXAI.wrap(model(), explain_plan=plan).explain_one(
            [0.4], object_id="bad-sha", dataset_version="fixture-v1")


def test_type2_audit_export_reopens_and_diagnostic_route_is_usable(tmp_path):
    import json
    from fuzzyxai.visualization.view_model import ExplanationViewModel

    result = FuzzyXAI.wrap(model()).explain_one(
        [0.4], object_id="durable-identity-001",
        dataset_version="synthetic-protocol-v1",
    )
    report = result.diagnose(repair_mode="none")
    assert isinstance(report.route_status, str)
    assert report.trace_sha256
    observed_guard = result.prediction.metadata["it2_guard"]
    path = tmp_path / "it2_result.json"
    result.export_json(path)
    reloaded = ExplanationViewModel.load_json(path)
    assert reloaded.risk["it2_guard"] == observed_guard
    assert reloaded.model["metadata"]["model_sha256"] == result.prediction.metadata["model_sha256"]
    assert json.loads(path.read_text(encoding="utf-8"))["risk"]["it2_guard"]["status"] == "verified"
