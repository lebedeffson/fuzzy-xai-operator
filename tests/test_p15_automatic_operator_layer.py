"""P16: scientific Γ/Δ/ρ (supersedes P15.7's heuristic version).

P15.7 computed Γ from a percent-of-supporting-claims-style proxy and Δ from
a surrogate-fidelity gap — neither is what the dissertation's chapter 2/3
actually define. This now uses the real machinery: Γ = semantic_disagreement
over genuine ExplanationObject pairs, Δ = the measured linear reconstruction
chain (P15.1), I_pre = compute_interpretability_index, and ρ = the real
5-component chapter-3 formula (predicted_risk, uncertainty,
interpretability_gap, reduction_loss, diagnostic) from
fuzzyxai.risk.risk_function.DEFAULT_RISK_WEIGHTS.

The key behavioral consequence: most single-channel sklearn models (linear,
tree, ensemble — only one local-explanation channel) genuinely have no
second explanatory object to compare against, so Γ correctly stays
unmeasured for them and E5 is *not* automatically reachable — this is more
conservative than P15.7, and correct. E5 requires either a real second
channel (e.g. a fuzzy/rule model with both native rule activations and a
numeric contribution channel) or manually supplied `evidence={"alignment":
...}`.
"""

from __future__ import annotations

from fuzzyxai import FuzzyXAI
from fuzzyxai.core.explain_plan import ExplainPlan
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


def _native_to_contribution_transform() -> dict[str, object]:
    return {
        "transform_id": "native-rule-to-feature-v1",
        "source_interface": "native_rules",
        "target_interface": "contribution",
        "term_mapping": {"mean radius:high": "mean radius"},
        "rule_mapping": {"R1": "R1"},
        "representation_mapping": "identity",
        "uncertainty_mapping": "identity",
        "trace_mapping": "identity",
        "source_refs": ["test-native-rule"],
        "limitations": ["Maps a declared rule term to its measured feature interface."],
    }


def test_single_channel_model_stays_at_the_honest_e4_ceiling() -> None:
    """A bare linear model has exactly one local-explanation channel
    (numeric contributions) — there is no second explanatory object for Γ
    to compare against automatically, so E5 must not be fabricated. With
    training history also supplied, E4 (one level below E5) is the correct,
    honest ceiling."""

    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, reference_data=X_train, reference_labels=y_train)
    history = {"objects": {"p0": [{"epoch": 1, "correct": True, "confidence": 0.9}, {"epoch": 2, "correct": False, "confidence": 0.4}]}}
    training_run = fx.observe_training(history=history)
    result = fx.explain_one(X_test[0], object_id="p0", include_counterfactuals=True, training_run=training_run, include_training_trace=True)
    assert result.explanation_level == "E4"
    assert result.view_model.disagreement["gamma"] is None
    # P18 item 1: a default ExplainPlan never declares alignment applicable
    # for a single-channel model — its absence is honestly not_applicable,
    # never "missing" (this model's plan never called for a second channel
    # in the first place).
    assert "alignment" not in result.missing_channels
    assert "alignment" in result.view_model.explanation_level["not_applicable_channels"]


def test_two_channel_model_reaches_e5_with_a_real_measured_gamma() -> None:
    """A model that genuinely supplies two explanatory channels (native
    fuzzy rule activations + numeric contributions for the same object)
    gets a real, measured Γ from semantic_disagreement — not a fabricated
    single-component proxy. E5 additionally needs a real, measured
    reduction (P17: no longer auto-derived from reconstruction fidelity),
    supplied here manually — exactly as honest as the automatic Γ."""

    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, reference_data=X_train, reference_labels=y_train)
    result = fx.explain_one(
        X_test[0],
        object_id="p0",
        include_counterfactuals=True,
        evidence={
            "activated_rules": [
                {"rule_id": "R1", "terms": [{"feature": "mean radius", "term": "high"}], "activation_strength": 0.8, "conclusion": "1"},
            ],
            "alignment": {"transform": _native_to_contribution_transform()},
            "reduction": {"components": {"term_loss": 0.05}, "weights": {"term_loss": 1.0}, "delta_max": 0.6},
        },
    )
    assert result.view_model.disagreement["gamma"] is not None
    assert set(result.view_model.disagreement["components"].keys()) == {"d_mu", "d_R", "d_alpha", "d_u", "d_tau", "d_L"}
    assert result.view_model.disagreement["delta"] == 0.05
    assert result.explanation_level == "E5"


def test_default_local_plan_does_not_export_a_partial_score_as_full_rho() -> None:
    """P19: a bare local explanation lacks non-zero-weight uncertainty and
    reduction inputs required by the dissertation's five-component rho.
    Its legacy subset remains available only as ``partial_risk_score`` and
    cannot authorize an automatic action under the name of full rho."""

    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model)
    result = fx.explain_one(X_test[0], object_id="p0")
    assert result.action == "insufficient_evidence"
    assert result.view_model.risk["status"] == "incomplete"
    assert result.view_model.risk["rho"] is None
    assert set(result.view_model.risk["missing_required_components"]) == {"uncertainty", "reduction_loss"}
    assert result.view_model.risk["partial_risk_score"] is not None


def test_auto_action_can_be_insufficient_evidence_when_a_declared_policy_has_no_real_source() -> None:
    """P18 item 1/3: when the resolved ExplainPlan genuinely declares an
    uncertainty source (here: ensemble_disagreement) but the wrapped model
    is not actually an ensemble and never supplies one, that component is a
    real gap, not a not_applicable one — the risk interface is disclosed as
    incomplete and action is demoted to "insufficient_evidence" rather than
    reading as a confident "accept"."""

    from fuzzyxai.core.explain_plan import UncertaintyPolicy

    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    plan = ExplainPlan(uncertainty_policy=UncertaintyPolicy(method="ensemble_disagreement", source="declared for this test"))
    fx = FuzzyXAI.wrap(model, explain_plan=plan)
    result = fx.explain_one(X_test[0], object_id="p0")
    assert result.action == "insufficient_evidence"
    assert result.view_model.risk["status"] == "incomplete"
    assert set(result.view_model.risk["missing_required_components"]) == {"uncertainty", "reduction_loss"}
    assert result.view_model.risk["rho"] is None
    assert result.view_model.risk["partial_risk_score"] is not None


def test_manual_evidence_still_overrides_automatic_computation() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model)
    result = fx.explain_one(
        X_test[0],
        object_id="p0",
        evidence={"risk": {"components": {"custom": 0.95}, "weights": {"custom": 1.0}}},
    )
    assert result.view_model.risk["components"] == {"custom": 0.95}
    assert result.action == "block"  # 0.95 > rho_critical=0.85 by default


def test_delta_is_not_conflated_with_reconstruction_fidelity() -> None:
    """P17: Δ (reduction loss) is a different quantity from linear
    reconstruction fidelity (|reconstructed_score - actual_score|) — the
    latter stays a standalone quality metric and does NOT automatically
    become Δ, since no real representation-reduction operation (Π) runs
    automatically for a linear model."""

    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model)
    result = fx.explain_one(X_test[0], object_id="p0")
    assert result.view_model.disagreement["delta"] is None
    assert result.view_model.disagreement["reduction_status"] == "not_applied"
    assert result.view_model.quality_metrics["reconstruction_error"] is not None
    assert result.view_model.quality_metrics["reconstruction_error"] < 1e-6


def test_delta_stays_not_applied_for_a_model_family_with_no_reduction_operation() -> None:
    from sklearn.tree import DecisionTreeClassifier

    X_train, X_test, y_train, _ = _split()
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model)
    result = fx.explain_one(X_test[0], object_id="p0")
    assert result.view_model.disagreement["delta"] is None
    assert result.view_model.disagreement["reduction_status"] == "not_applied"


def test_no_score_and_no_fidelity_signal_falls_back_to_missing_not_fabricated() -> None:
    """When there is genuinely no signal (no score, no contribution method
    at all), risk must stay unmeasured rather than substituting 0."""

    from fuzzyxai.adapters.contracts_v2 import ExplanationContext, LocalModelEvidence
    from fuzzyxai.adapters.model import ModelPrediction
    from fuzzyxai.adapters.model_v2 import ModelAdapterV2

    class BareAdapter(ModelAdapterV2):
        adapter_id = "bare"
        model_family = "bare"

        def predict(self, inputs):
            return ModelPrediction(predictions=[0], probabilities=None, model_type="bare", adapter_id=self.adapter_id, metadata={"task_type": self.task_type.value})

        def extract_local_evidence(self, inputs, prediction, context: ExplanationContext) -> LocalModelEvidence:
            return LocalModelEvidence(channels={})

        def feature_names(self):
            return ["a"]

        def model_fingerprint(self):
            return "0" * 16

    fx = FuzzyXAI.wrap(object(), adapter=BareAdapter(object(), task="classification"))
    result = fx.explain_one([1.0], object_id="p0")
    assert "risk" in result.missing_channels
    assert result.view_model.risk["rho"] is None


def test_explain_plan_metadata_can_override_risk_weights() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    plan = ExplainPlan(metadata={"risk_weights": {"predicted_risk": 1.0}})  # drop every other component entirely, including diagnostic
    fx = FuzzyXAI.wrap(model, explain_plan=plan)
    result = fx.explain_one(X_test[0], object_id="p0")
    assert set(result.view_model.risk["components"].keys()) == {"predicted_risk"}


def test_random_forest_does_not_relabel_global_importance_as_local_evidence() -> None:
    """RF vote disagreement is native uncertainty, but global feature
    importance cannot supply a local interpretability channel."""

    from fuzzyxai.core.explain_plan import UncertaintyPolicy

    X_train, X_test, y_train, _ = _split()
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X_train, y_train)
    plan = ExplainPlan(uncertainty_policy=UncertaintyPolicy(method="ensemble_disagreement", source="RandomForestClassifier per-tree vote disagreement"))
    fx = FuzzyXAI.wrap(model, explain_plan=plan)
    result = fx.explain_one(X_test[0], object_id="p0")
    assert result.view_model.model["contributions"] == {}
    assert "uncertainty" in result.view_model.risk["components"]
    assert result.view_model.risk["status"] == "incomplete"
    assert result.view_model.risk["rho"] is None
    assert result.action == "insufficient_evidence"


def test_critical_structural_failure_forces_block_despite_low_rho() -> None:
    """Section 6's acceptance criterion: a critical explanatory rupture
    (Γ exceeding gamma_critical) must force "block" even when every other
    risk component looks good — good numbers cannot buy back a broken
    explanation chain."""

    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    plan = ExplainPlan(gamma_critical=0.02)  # force a near-certain alignment failure
    fx = FuzzyXAI.wrap(model, explain_plan=plan)
    result = fx.explain_one(
        X_test[0],
        object_id="p0",
        evidence={
            "activated_rules": [{"rule_id": "R1", "terms": [{"feature": "mean radius", "term": "high"}], "activation_strength": 0.9, "conclusion": "1"}],
            "alignment": {"transform": _native_to_contribution_transform()},
        },
    )
    assert result.view_model.disagreement["gamma"] is not None
    assert result.view_model.risk["chi_r_crit"] == 1
    assert result.action == "block"
