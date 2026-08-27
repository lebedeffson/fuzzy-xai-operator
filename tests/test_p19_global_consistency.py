from __future__ import annotations

from dataclasses import replace
import inspect
import numpy as np
import fuzzyxai
import pytest
import fuzzyxai.runtime as runtime_module
from fuzzyxai.adapters.contracts_v2 import ExplanationContext
from fuzzyxai.adapters.model import ModelPrediction
from fuzzyxai.adapters.sklearn_v2 import SklearnEnsembleAdapter
from fuzzyxai.adapters.system_source import derive_system_source_evidence
from fuzzyxai.core.explain_plan import (
    AlignmentPolicy,
    ExplainPlan,
    MembershipPolicy,
    MembershipTerm,
    ReductionPolicy,
    UncertaintyPolicy,
    UncertaintyRepresentationPolicy,
)
from fuzzyxai.core.explanation_object import Rule, Trace
from fuzzyxai.core.external_tabular_route import build_external_wine_classification_route
from fuzzyxai.core.types import AdaptedInput
from fuzzyxai.evidence import ExplanationEvidence, determine_explanation_level
from fuzzyxai.scientific_alignment import AlignmentTransform
from fuzzyxai.system_semantics import SystemObservation, SystemSourceEvidence, build_system_evidence
from fuzzyxai.visualization import shap_like
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from final_transparency_validation.generate_p19_system_cases import _system_plan


def _membership() -> MembershipPolicy:
    return MembershipPolicy(
        "malignant_risk",
        (0.0, 1.0),
        (
            MembershipTerm("low", "triangular", (0.0, 0.0, 0.5)),
            MembershipTerm("medium", "triangular", (0.2, 0.5, 0.8)),
            MembershipTerm("high", "triangular", (0.5, 1.0, 1.0)),
        ),
        "preset",
        "test-v1",
    )


def _transform(identifier: str = "test-transform") -> AlignmentTransform:
    return AlignmentTransform(
        identifier,
        "class_probability",
        "risk_membership",
        {"class_probability:class:0": "risk:high", "class_probability:class:1": "risk:low"},
        {"source_class_0": "target_high", "source_class_1": "target_low"},
        "risk_membership_partition",
        parameters={
            "triangles": {
                "low": [0.0, 0.0, 0.5],
                "medium": [0.2, 0.5, 0.8],
                "high": [0.5, 1.0, 1.0],
            }
        },
    )


def _plan(
    *,
    alignment: bool = True,
    reduction: bool = True,
    uncertainty: str = "ensemble_vote_standard_deviation",
    transform: AlignmentTransform | None = None,
) -> ExplainPlan:
    transform = transform or _transform()
    membership = _membership()
    return ExplainPlan(
        gamma_critical=0.6,
        rho_accept=0.35,
        rho_warning=0.60,
        rho_audit=0.85,
        rho_critical=0.95,
        alignment_policy=AlignmentPolicy(
            applicable=alignment,
            source="test",
            transform=transform.to_dict() if alignment else {},
        ),
        reduction_policy=ReductionPolicy(
            applicable=reduction,
            method="F_int_to_F0_midpoint" if reduction else "none",
            source="test",
        ),
        uncertainty_policy=UncertaintyPolicy(method=uncertainty, source="test"),
        uncertainty_representation_policy=UncertaintyRepresentationPolicy(),
        membership_policies={"system_risk": membership},
        metadata={
            "system_risk_weights": {
                "w_p": 0.30,
                "w_u": 0.25,
                "w_I": 0.20,
                "w_Delta": 0.15,
                "w_R": 0.10,
            },
        },
    )


def _observation(transform: AlignmentTransform | None = None) -> SystemObservation:
    trace = Trace(
        "object",
        "risk-v1",
        "2026-01-01T00:00:00+00:00",
        source="test-verifier",
        checksum="test-checksum",
    )
    return SystemObservation(
        transform or _transform(),
        _membership(),
        1,
        trace,
        model_trace=trace,
        trace_verification_source="test verifier",
    )


def _build(plan: ExplainPlan, observation: SystemObservation | None = None):
    trace = (observation or _observation()).model_trace
    assert trace is not None
    source = SystemSourceEvidence(
        source_interface_id="class_probability",
        terms=("class_probability:class:0", "class_probability:class:1"),
        representation_value=0.25,
        representation_label="class_probability:1",
        rules=(
            Rule("source_class_0", {"model_class": "0"}, "0"),
            Rule("source_class_1", {"model_class": "1"}, "1"),
        ),
        activations={"source_class_0": 0.25, "source_class_1": 0.75},
        model_uncertainty_inputs={
            "ensemble_vote_standard_deviation": 0.4330127018922193,
            "probabilities": [0.25, 0.75],
        },
        trace=trace,
        source_refs=("test source",),
    )
    return build_system_evidence(
        object_id="object",
        model_fingerprint="model-fingerprint",
        source=source,
        plan=plan,
        observation=observation or _observation(),
    )


class _FixedVoteEstimator:
    def __init__(self, label: object):
        self.label = label

    def predict(self, inputs: object) -> np.ndarray:
        del inputs
        return np.asarray([self.label], dtype=object)


class _FixedVoteEnsemble:
    _estimator_type = "classifier"

    def __init__(self, labels: tuple[object, object], votes: tuple[object, ...]):
        self.classes_ = np.asarray(labels, dtype=object)
        self.estimators_ = [_FixedVoteEstimator(label) for label in votes]


@pytest.mark.parametrize(
    ("labels", "votes", "risk_class"),
    [
        ((0, 1), (0, 0, 0, 1), 1),
        ((-1, 1), (-1, -1, -1, 1), 1),
        (("safe", "risk"), ("safe", "safe", "safe", "risk"), "risk"),
    ],
)
def test_rf_vote_uncertainty_is_invariant_to_class_encoding(
    labels: tuple[object, object], votes: tuple[object, ...], risk_class: object,
) -> None:
    model = _FixedVoteEnsemble(labels, votes)
    adapter = SklearnEnsembleAdapter(model)
    prediction = ModelPrediction(
        predictions=[labels[0]], probabilities=[[0.75, 0.25]],
        metadata={"classes": list(labels)},
    )
    local = adapter.extract_local_evidence(np.asarray([[0.0]]), prediction, ExplanationContext())
    trace = _observation().trace
    source = derive_system_source_evidence(
        object_id="encoded-labels", model_fingerprint="fixed-vote",
        prediction=prediction, internal_evidence=local.channels,
        source_interface_id="class_probability", risk_class=risk_class,
        trace=trace, model_trace=trace,
    )
    expected = pytest.approx(np.std([0.0, 0.0, 0.0, 1.0]))
    assert local.channels["ensemble_disagreement"] == expected
    assert source.model_uncertainty_inputs["ensemble_vote_standard_deviation"] == expected
    assert source.model_uncertainty_inputs["vote_indicator_risk_class"] == risk_class


def test_rf_probability_is_not_replaced_by_hard_vote_proportion() -> None:
    trace = _observation().trace
    prediction = ModelPrediction(
        predictions=[1], probabilities=[[0.30, 0.70]],
        metadata={"classes": [0, 1]},
    )
    source = derive_system_source_evidence(
        object_id="probability-vs-votes", model_fingerprint="fixed-vote",
        prediction=prediction,
        internal_evidence={"ensemble_votes": [[0], [0], [0], [1]]},
        source_interface_id="class_probability", risk_class=1,
        trace=trace, model_trace=trace,
    )
    assert source.representation_value == pytest.approx(0.70)
    assert source.metadata is not None
    assert source.metadata["representation_semantics"] == "class_probability"
    assert source.model_uncertainty_inputs["vote_proportions"]["1"] == pytest.approx(0.25)


def test_vote_only_fallback_discloses_vote_proportion_semantics() -> None:
    trace = _observation().trace
    source = derive_system_source_evidence(
        object_id="vote-fallback", model_fingerprint="fixed-vote",
        prediction=ModelPrediction(predictions=[1], metadata={"classes": [0, 1]}),
        internal_evidence={"ensemble_votes": [[0], [0], [0], [1]]},
        source_interface_id="class_probability", risk_class=1,
        trace=trace, model_trace=trace,
    )
    assert source.representation_value == pytest.approx(0.25)
    assert source.representation_label == "vote_proportion:1"
    assert source.metadata is not None
    assert source.metadata["representation_semantics"] == "vote_proportion"


def test_noncritical_rupture_contributes_to_rho_without_forcing_block() -> None:
    observation = replace(
        _observation(), rupture_present=True, critical_rupture=False,
        rupture_code="test_noncritical_rupture",
    )
    system = _build(_plan(), observation)
    assert system.risk.components["chi_R"] == 1.0
    assert system.risk.chi_R_critical == 0
    assert system.risk.critical_override is False
    assert system.risk.rho is not None
    assert system.risk.action == system.risk.candidate_action
    assert system.risk.rho >= system.risk.weights["chi_R"]
    assert system.diagnostics[0]["severity"] == "warning"


def test_critical_rupture_keeps_numeric_rho_and_blocks() -> None:
    observation = replace(
        _observation(), rupture_present=True, critical_rupture=True,
        rupture_code="test_critical_rupture",
    )
    system = _build(_plan(), observation)
    assert system.risk.components["chi_R"] == 1.0
    assert system.risk.chi_R_critical == 1
    assert system.risk.rho is not None
    assert system.risk.critical_override is True
    assert system.risk.action == "block"


def test_public_system_view_ignores_zero_weight_missing_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_build = runtime_module.build_system_evidence

    def build_with_zero_weight_missing(**kwargs: object):
        system = real_build(**kwargs)
        components = {**system.risk.components, "u_M": None}
        weights = {**system.risk.weights, "rho_p": system.risk.weights["rho_p"] + system.risk.weights["u_M"], "u_M": 0.0}
        complete = replace(system.risk, components=components, weights=weights, status="complete")
        return replace(system, risk=complete)

    monkeypatch.setattr(runtime_module, "build_system_evidence", build_with_zero_weight_missing)
    X, y = make_classification(n_samples=80, n_features=6, n_informative=4, random_state=23)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    transform = _transform("public-zero-weight-transform")
    trace = Trace("public-zero", "logistic-v1", "2026-01-01T00:00:00+00:00", source="test", checksum="same-model")
    context = fuzzyxai.ObservationContext(
        reference_data=X, reference_labels=y,
        system_observation=SystemObservation(transform, _membership(), 1, trace, model_trace=trace),
    )
    result = fuzzyxai.FuzzyXAI.wrap(
        model, explain_plan=_plan(uncertainty="entropy", transform=transform),
        observation_context=context,
    ).explain_one(X[0], object_id="public-zero")
    assert result.system is not None
    assert result.system.risk.status == "complete"
    assert result.system.risk.rho is not None
    assert result.view_model.risk["missing_required_components"] == []
    assert result.audit()["system_evidence"]["risk"]["status"] == "complete"
    assert result.to_dict(detail="audit")["risk"]["missing_required_components"] == []


def test_complete_system_risk_has_no_stale_missing_components() -> None:
    system = _build(_plan())
    assert system.risk.status == "complete"
    assert system.risk.rho is not None
    assert [name for name, value in system.risk.components.items() if value is None] == []
    assert set(system.risk.components) == {
        "rho_p", "u_M", "one_minus_I_pre", "Delta", "chi_R",
    }


def test_rf_vote_evidence_does_not_imply_local_contributions() -> None:
    level = determine_explanation_level(
        ExplanationEvidence(),
        contribution_method=None,
        operator_channels={"alignment": True, "reduction": True, "risk": True},
        local_contributions_supported=False,
    )
    assert "local_contributions" not in level.available_channels
    assert level.channel_status["local_contributions"] == "not_applicable"


def test_optional_missing_is_not_required_missing() -> None:
    level = determine_explanation_level(
        ExplanationEvidence(),
        contribution_method=None,
        operator_channels={"alignment": True, "reduction": True, "risk": True},
        required_channels=("prediction", "call_trace", "risk"),
    )
    assert "training_history" in level.optional_missing_channels
    assert "counterfactuals" in level.optional_missing_channels
    assert "training_history" not in level.required_missing_channels
    assert level.channel_status["training_history"] == "optional_missing"


def test_explain_plan_alignment_applicability_is_respected() -> None:
    with pytest.raises(ValueError, match="not applicable"):
        _build(_plan(alignment=False))


def test_explain_plan_reduction_not_applicable_is_not_executed() -> None:
    system = _build(_plan(reduction=False))
    assert system.reduction is None
    assert system.reduction_status == "not_applied"
    assert system.risk.components["Delta"] == 0.0


def test_explain_plan_uncertainty_method_is_respected() -> None:
    measured = _build(_plan(uncertainty="ensemble_disagreement"))
    assert measured.uncertainty.sources["U_model"]["method"] == "ensemble_vote_standard_deviation"
    with pytest.raises(ValueError, match="uncertainty method 'none'"):
        _build(_plan(uncertainty="none"))


def test_mismatching_plan_and_observation_transform_fails_closed() -> None:
    with pytest.raises(ValueError, match="conflicts with ExplainPlan"):
        _build(_plan(transform=_transform("plan-transform")), _observation(_transform("observation-transform")))


def test_accept_and_conflict_share_one_system_policy() -> None:
    transform = _transform()
    membership = _membership()
    accept_plan = _system_plan(transform, membership)
    conflict_plan = _system_plan(transform, membership)
    assert accept_plan.to_dict() == conflict_plan.to_dict()
    assert accept_plan.gamma_critical == 0.60


def test_non_rf_probability_model_uses_same_public_system_runtime() -> None:
    X, y = make_classification(n_samples=80, n_features=6, n_informative=4, random_state=19)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    transform = _transform("logistic-probability-transform")
    plan = _plan(uncertainty="entropy", transform=transform)
    trace = Trace("non-rf", "logistic-v1", "2026-01-01T00:00:00+00:00", source="sklearn.LogisticRegression", checksum="same-model")
    context = fuzzyxai.ObservationContext(
        reference_data=X,
        reference_labels=y,
        system_observation=SystemObservation(
            transform,
            _membership(),
            1,
            trace,
            model_trace=trace,
            trace_verification_source="test same-model verifier",
        ),
    )
    result = fuzzyxai.FuzzyXAI.wrap(model, explain_plan=plan, observation_context=context).explain_one(X[0], object_id="non-rf")
    assert result.system is not None
    assert result.system.source_evidence.metadata is not None
    assert result.system.source_evidence.metadata["provider"] == "native_class_probabilities"
    assert result.system.uncertainty.sources["U_model"]["method"] == "entropy"
    assert result.system.risk.status == "complete"


def test_public_canonical_risk_is_strict_five_component_contract() -> None:
    components = {
        "rho_p": 0.1,
        "u_M": 0.2,
        "one_minus_I_pre": 0.3,
        "Delta": 0.4,
        "chi_R": 0.0,
    }
    weights = {
        "rho_p": 0.3,
        "u_M": 0.25,
        "one_minus_I_pre": 0.2,
        "Delta": 0.15,
        "chi_R": 0.1,
    }
    result = fuzzyxai.observe_risk(
        components,
        weights,
        {"theta_1": 0.1, "theta_2": 0.3, "theta_3": 0.6, "theta_4": 0.8},
        0,
    )
    assert result.rho == pytest.approx(0.20)
    assert result.action == "lower_confidence"
    with pytest.raises(ValueError, match="requires exactly"):
        fuzzyxai.observe_risk(
            {"uncertainty": 0.2}, {"uncertainty": 1.0},
            {"theta_1": 0.1, "theta_2": 0.3, "theta_3": 0.6, "theta_4": 0.8},
            0,
        )


def test_canonical_visualization_does_not_label_max_gamma_delta_as_rho() -> None:
    source = inspect.getsource(shap_like.render_gamma_delta_action_map_v2)
    assert "rho=max" not in source.replace(" ", "")
    assert "legacy_route_score" in inspect.getsource(shap_like.local_risk_evidence_data)


def test_external_compatibility_route_does_not_export_p19_names() -> None:
    route = build_external_wine_classification_route(
        AdaptedInput(
            scenario_id="external_wine_classification",
            values={
                "source_type": "external",
                "model_name": "external model",
                "dataset_name": "external data",
                "predicted_class": 1,
                "class_probability": 0.8,
                "feature_values": {"a": 1.0, "b": 2.0},
                "feature_importance": {"a": 0.6, "b": 0.2},
                "quality_metrics": {},
            },
        )
    )
    assert not ({"gamma", "delta", "rho"} & set(route.computed_result))
    assert route.computed_result["presentation_omission_loss"] == pytest.approx(0.2)
    assert route.computed_result["scientific_contract"] == "legacy_external_route_metrics_not_P19_Gamma_Delta_rho"
