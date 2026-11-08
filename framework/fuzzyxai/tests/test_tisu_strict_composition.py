"""Contract, orientation and distance tests for opt-in strict TISU operator."""
from __future__ import annotations

from itertools import product

from fuzzyxai import ExplainPlan, Rule, SystemOperator, Trace
from fuzzyxai.core.strict_composition import (
    RegisteredAlignment,
    compose_with_registered_alignment,
    conditional_explanation_distance,
)

WEIGHTS = {"repr": 0.0, "rules": 0.05, "activations": 0.6, "uncertainty": 0.1,
           "trace": 0.05, "reduction": 0.1, "gamma_max": 1.0}
VOCAB = ("r_high", "r2", "r3")


def _obj(**changes):
    plan = ExplainPlan()
    obj = SystemOperator(plan).explain_scalar_risk(
        0.7, [Rule("r_high", {"risk": "high"}, "check")],
        Trace("sample-1", "v1", "time", source="source", checksum="hash"),
    )
    contract = {
        "sample_id": "sample-1", "model_fingerprint": "model-v1",
        "feature_schema": "schema-v1", "event_id": "disease",
        "probability": 0.7, "provenance_ref": "source-hash",
    }
    contract.update(changes)
    obj.metadata["interface"] = contract
    return obj


def _compose(a, b, alignment=None):
    return compose_with_registered_alignment(a, b, WEIGHTS, VOCAB, alignment)


def _reject(value):
    assert getattr(value, "code", None) == "D_ij"


def test_matching_declarations_compose():
    result = _compose(_obj(), _obj())
    assert getattr(result, "code", None) != "D_ij"
    assert result.metadata["semantic_truth"] == "UNVERIFIED_EXTERNAL"


def test_opposite_events_require_explicit_transform():
    a, b = _obj(), _obj(event_id="healthy", probability=0.3)
    _reject(_compose(a, b))
    result = _compose(a, b, RegisteredAlignment("disease", "healthy", "complement"))
    assert getattr(result, "code", None) != "D_ij"


def test_wrong_complement_is_rejected():
    _reject(_compose(
        _obj(), _obj(event_id="healthy", probability=0.7),
        RegisteredAlignment("disease", "healthy", "complement"),
    ))


def test_cross_sample_schema_or_model_is_rejected():
    for change in ({"sample_id": "different"}, {"feature_schema": "other"},
                   {"model_fingerprint": "different"}):
        _reject(_compose(_obj(), _obj(**change)))


def test_absent_provenance_ref_is_rejected():
    _reject(_compose(_obj(), _obj(provenance_ref="")))


def test_conflicting_rule_identifier_is_rejected():
    a, b = _obj(), _obj()
    b.rules = [Rule("r_high", {"risk": "high"}, "deny")]
    _reject(_compose(a, b))


def test_declared_renaming_can_compose():
    a, b = _obj(), _obj()
    b.terms = {"low-risk", "middle-risk", "high-risk"}
    _reject(_compose(a, b))
    transform = RegisteredAlignment(
        "disease", "disease", "identity",
        term_map={"low": "low-risk", "medium": "middle-risk", "high": "high-risk"},
    )
    assert getattr(_compose(a, b, transform), "code", None) != "D_ij"


def test_nonbijective_term_mapping_is_rejected():
    a, b = _obj(), _obj()
    b.terms = {"low-risk", "middle-risk"}
    transform = RegisteredAlignment(
        "disease", "disease", "identity",
        term_map={"low": "low-risk", "medium": "low-risk", "high": "middle-risk"},
    )
    _reject(_compose(a, b, transform))


def test_unknown_activation_coordinate_is_rejected():
    a = _obj()
    a.activations["not-in-vocabulary"] = 1.0
    _reject(_compose(a, _obj()))


def test_nonzero_reduction_has_zero_self_distance():
    a = _obj().copy_with(reduction_loss=0.2)
    assert conditional_explanation_distance(a, a, WEIGHTS, VOCAB) == 0.0


def test_metric_triangle_on_small_finite_grid():
    states = []
    for values in product((0.0, 0.5, 1.0), repeat=3):
        x = _obj().copy_with(
            activations=dict(zip(VOCAB, values)), rules=[],
        )
        states.append(x)
    for a, b, c in product(states, repeat=3):
        assert conditional_explanation_distance(a, c, WEIGHTS, VOCAB) <= (
            conditional_explanation_distance(a, b, WEIGHTS, VOCAB)
            + conditional_explanation_distance(b, c, WEIGHTS, VOCAB)
            + 1e-12
        )


def test_unregistered_representation_distance_rejected():
    weights = dict(WEIGHTS, repr=0.3)
    value = compose_with_registered_alignment(_obj(), _obj(), weights, VOCAB)
    _reject(value)
