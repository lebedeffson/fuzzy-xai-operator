"""Wheel-only P19 smoke: run outside the source checkout."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib import metadata, util

import numpy as np
from fuzzyxai import AlignmentTransform, ExplainPlan, FuzzyXAI, ObservationContext, SystemObservation, Trace
from fuzzyxai.audit.operators_manifest import validate_manifest
from fuzzyxai.core.explain_plan import AlignmentPolicy, MembershipPolicy, MembershipTerm, ReductionPolicy, UncertaintyPolicy, UncertaintyRepresentationPolicy
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    manifest = validate_manifest()
    assert manifest["status"] == "PASS", manifest
    assert manifest["reference_files_verified"] is False, manifest
    requires = metadata.requires("fuzzyxai-operator") or []
    assert not any(item.lower().startswith("nicegui") and "extra ==" not in item.lower() for item in requires), requires
    assert util.find_spec("nicegui") is None
    for research_package in (
        "fuzzyxai.q1_validation",
        "fuzzyxai.q1_final",
        "fuzzyxai.final_closure",
        "fuzzyxai.ai_pre_review",
        "fuzzyxai.ai_pre_review_final",
        "fuzzyxai.strong_confirmatory",
    ):
        assert util.find_spec(research_package) is None, research_package

    data = load_breast_cancer()
    x_train, x_test, y_train, _ = train_test_split(data.data, data.target, test_size=0.2, random_state=0, stratify=data.target)
    tabular = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000))]).fit(x_train, y_train)
    local = FuzzyXAI.wrap(tabular, observation_context=ObservationContext(reference_data=x_train, reference_labels=y_train)).explain_one(x_test[0])
    assert local.prediction.predictions
    assert local.full_report(level="reader")
    assert local.audit()["graph"]
    assert local.to_dict(detail="audit")["model"]
    assert local.inspect("action").target_id == "action"

    forest = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=0).fit(x_train, y_train)
    index = int(np.argmin(forest.predict_proba(x_test)[:, list(forest.classes_).index(0)]))
    transform = AlignmentTransform(
        "wheel-class-probability-to-risk-v2",
        "class_probability",
        "risk_membership",
        {"class_probability:class:0": "risk:high", "class_probability:class:1": "risk:low"},
        {"source_class_0": "target_high", "source_class_1": "target_low"},
        "risk_membership_partition",
        parameters={"triangles": {"low": [0.0, 0.0, 0.5], "medium": [0.2, 0.5, 0.8], "high": [0.5, 1.0, 1.0]}},
        source_refs=("RandomForestClassifier.predict", "wheel-smoke-risk-policy"),
    )
    membership = MembershipPolicy(
        "malignant_risk",
        (0.0, 1.0),
        (
            MembershipTerm("low", "triangular", (0.0, 0.0, 0.5)),
            MembershipTerm("medium", "triangular", (0.2, 0.5, 0.8)),
            MembershipTerm("high", "triangular", (0.5, 1.0, 1.0)),
        ),
        "preset",
        "wheel-smoke-v1",
    )
    stamp = datetime.now(UTC).isoformat()
    trace = Trace("wheel-object", "rf-vote-v1", stamp, source="RandomForestClassifier.estimators_", checksum="wheel-smoke")
    plan = ExplainPlan(
        rho_accept=0.35,
        rho_warning=0.60,
        rho_audit=0.85,
        rho_critical=0.95,
        alignment_policy=AlignmentPolicy(True, "wheel smoke", transform.to_dict()),
        reduction_policy=ReductionPolicy(True, "F_int_to_F0_midpoint", "wheel smoke"),
        uncertainty_policy=UncertaintyPolicy("ensemble_vote_standard_deviation", source="native per-tree predictions"),
        uncertainty_representation_policy=UncertaintyRepresentationPolicy(source="wheel smoke declared heuristic; not calibrated"),
        membership_policies={"system_risk": membership},
        metadata={
            "system_risk_weights": {"w_p": 0.3, "w_u": 0.25, "w_I": 0.2, "w_Delta": 0.15, "w_R": 0.1},
            "system_action_policy": {"theta_2_to_theta_3": "request_more_data", "theta_3_to_theta_4": "defer_to_human"},
            "system_target_consequents": {"low": "accept", "medium": "review", "high": "defer_to_human"},
        },
    )
    context = ObservationContext(
        reference_data=x_train,
        reference_labels=y_train,
        dataset_version="wheel-smoke-bcw",
        system_observation=SystemObservation(transform, membership, 0, trace, model_trace=trace, trace_complete=True),
    )
    system = FuzzyXAI.wrap(forest, explain_plan=plan, observation_context=context).explain_one(x_test[index], object_id="wheel-object")
    assert system.system is not None
    assert system.system.risk.status == "complete"
    assert system.system.risk.action == "accept"
    assert system.system.risk.candidate_action == "accept"
    assert system.system.risk.critical_override is False
    assert system.system.risk.chi_R_critical == 0
    assert system.system.source_evidence.metadata is not None
    assert system.system.source_evidence.metadata["representation_semantics"] == "class_probability"
    assert system.system.source_evidence.metadata["representation_source"] == "prediction.probabilities"
    assert system.system.uncertainty.sources["U_model"]["formula"] == "std(binary per-tree votes)"
    rho_inputs = {
        edge.source
        for edge in system.explanation_graph.edges
        if edge.target == "system:rho"
    }
    assert rho_inputs == {
        "system:rho_p",
        "system:u_M",
        "system:one_minus_I_pre",
        "system:Delta",
        "system:chi_R",
    }
    assert system.inspect("action").target_id == "action"

    probability_transform = AlignmentTransform(
        "wheel-logistic-probability-to-risk-v1",
        "class_probability",
        "risk_membership",
        {"class_probability:class:0": "risk:high", "class_probability:class:1": "risk:low"},
        {"source_class_0": "target_high", "source_class_1": "target_low"},
        "risk_membership_partition",
        parameters=transform.parameters,
        source_refs=("Pipeline.predict_proba", "wheel-smoke-risk-policy"),
    )
    probability_plan = ExplainPlan(
        rho_accept=0.35,
        rho_warning=0.60,
        rho_audit=0.85,
        rho_critical=0.95,
        alignment_policy=AlignmentPolicy(True, "wheel non-RF smoke", probability_transform.to_dict()),
        reduction_policy=ReductionPolicy(True, "F_int_to_F0_midpoint", "wheel non-RF smoke"),
        uncertainty_policy=UncertaintyPolicy("entropy", source="Pipeline.predict_proba"),
        uncertainty_representation_policy=UncertaintyRepresentationPolicy(source="wheel non-RF declared heuristic; not calibrated"),
        membership_policies={"system_risk": membership},
        metadata=plan.metadata,
    )
    probability_trace = Trace("wheel-logistic-object", "logistic-probability-v1", stamp, source="Pipeline.predict_proba", checksum="wheel-logistic-smoke")
    probability_context = ObservationContext(
        reference_data=x_train,
        reference_labels=y_train,
        dataset_version="wheel-smoke-bcw",
        system_observation=SystemObservation(probability_transform, membership, 0, probability_trace, model_trace=probability_trace, trace_complete=True),
    )
    non_rf = FuzzyXAI.wrap(tabular, explain_plan=probability_plan, observation_context=probability_context).explain_one(x_test[0], object_id="wheel-logistic-object")
    assert non_rf.system is not None
    assert non_rf.system.source_evidence.metadata is not None
    assert non_rf.system.source_evidence.metadata["provider"] == "native_class_probabilities"
    assert non_rf.system.risk.status == "complete"
    print({"manifest": manifest["status"], "operators": manifest["operator_count"], "local_action": local.action, "rf_system_action": system.system.risk.action, "rf_system_rho": system.system.risk.rho, "non_rf_provider": non_rf.system.source_evidence.metadata["provider"], "non_rf_action": non_rf.system.risk.action, "nicegui_installed": False, "research_packages_installed": False})


if __name__ == "__main__":
    main()
