"""Contract tests for independently anchored, sequential XAI composition."""

from dataclasses import replace
import hashlib

import pytest

from fuzzyxai.core.certified_route import (
    RouteStatus, SignedPrediction, sign_prediction, verify_prediction,
    certify_route,
)
from fuzzyxai.core.explanation_object import ExplanationObject, Rule, Trace


SECRET = b"research-demo-only-rotate-key-2026"
BETA = {
    "repr": 0.2, "rules": 0.2, "activations": 0.2,
    "uncertainty": 0.15, "trace": 0.15, "reduction": 0.1,
}


class ScalarRep:
    def __init__(self, value: float):
        self.value = value

    def distance(self, other):
        return abs(self.value - other.value)


def receipt(score: float = 0.72) -> SignedPrediction:
    return sign_prediction(SignedPrediction(
        object_id="sample-1", event_id="malignant", model_version="v1",
        unit="probability", score=score,
        source_sha256=hashlib.sha256(b"actual ordered features").hexdigest(),
        model_sha256=hashlib.sha256(b"fitted model bytes").hexdigest(),
        trace_checksum="trusted-original-trace",
        complement_event_id="benign",
    ), secret=SECRET)


def stage(a: SignedPrediction, *, event: str = "malignant", p: float | None = None,
          number: int = 0, **change):
    contract = {
        "object_id": a.object_id, "event_id": event, "model_version": a.model_version,
        "unit": a.unit, "source_sha256": a.source_sha256,
        "model_sha256": a.model_sha256,
    }
    contract.update(change)
    p = a.score if p is None else p
    return ExplanationObject(
        terms={"risk", "high"}, representation=ScalarRep(p),
        rules=[Rule(f"rule-{number}", {"risk": "high"}, "review")],
        activations={f"rule-{number}": 0.6}, uncertainty=0.12,
        trace=Trace(a.object_id, a.model_version, "2026-11-14", source="trusted-adapter",
                    checksum=a.trace_checksum if number == 0 else f"derived-{number}"),
        metadata={"semantic_contract": contract, "risk": p},
    )


def check(a, steps, **opts):
    return certify_route(steps, signed_prediction=a, verification_secret=SECRET, beta=BETA, **opts)


def test_clean_three_stage_route_and_fingerprint_stability():
    a = receipt()
    route = [stage(a, number=i) for i in range(3)]
    one, two = check(a, route), check(a, route)
    assert one.status == RouteStatus.CERTIFIED
    assert one.action == "eligible_for_policy_review"
    assert len(one.edge_gamma) == 2
    assert one.route_fingerprint == two.route_fingerprint
    assert len(one.route_fingerprint) == 64
    assert route[1].metadata["risk"] == a.score


def test_valid_binary_complement_and_explicit_mapping():
    a = receipt(0.77)
    left = stage(a)
    right = stage(a, event="benign", p=0.23, number=1)
    right.metadata["semantic_transition"] = {
        "kind": "complement", "from_event": "malignant", "to_event": "benign",
    }
    assert check(a, [left, right]).status == RouteStatus.CERTIFIED
    right.metadata["semantic_transition"]["from_event"] = "benign"
    assert check(a, [left, right]).status == RouteStatus.REJECTED


@pytest.mark.parametrize("attack", ["event", "score", "model", "source", "trace"])
def test_coherent_pairwise_spoof_cannot_change_signed_origin(attack):
    a = receipt()
    l, r = stage(a), stage(a, number=1)
    if attack == "event":
        for item in (l, r):
            item.metadata["semantic_contract"]["event_id"] = "benign"
    elif attack == "score":
        for item in (l, r):
            item.metadata["risk"] = 0.33
    elif attack == "model":
        for item in (l, r):
            item.metadata["semantic_contract"]["model_sha256"] = "f" * 64
    elif attack == "source":
        for item in (l, r):
            item.metadata["semantic_contract"]["source_sha256"] = "f" * 64
    else:
        l.trace = replace(l.trace, checksum="changed")
    result = check(a, [l, r])
    assert result.status == RouteStatus.REJECTED
    assert result.action == "block"


def test_complement_requires_independently_signed_event_pair():
    a = receipt()
    left = stage(a)
    arbitrary = stage(a, event="unrelated", p=1-a.score, number=1)
    arbitrary.metadata["semantic_transition"] = {
        "kind": "complement", "from_event": a.event_id, "to_event": "unrelated",
    }
    assert check(a, [left, arbitrary]).findings[0].code == "COMPLEMENT_NOT_AUTHORIZED"
    a_without_pair = sign_prediction(replace(a, complement_event_id=None, signature=""), secret=SECRET)
    valid_name_but_unsigned = stage(a_without_pair, event="benign", p=1-a.score, number=1)
    valid_name_but_unsigned.metadata["semantic_transition"] = {
        "kind": "complement", "from_event": "malignant", "to_event": "benign",
    }
    assert check(a_without_pair, [stage(a_without_pair), valid_name_but_unsigned]).findings[0].code == "COMPLEMENT_NOT_AUTHORIZED"


def test_scalar_representation_and_policy_are_bound_to_certificate():
    a = receipt()
    st = [stage(a), stage(a, number=1)]
    first = check(a, st)
    assert first.status == RouteStatus.CERTIFIED
    second = check(a, st, score_tolerance=1e-6)
    assert second.status == RouteStatus.CERTIFIED
    assert first.route_fingerprint != second.route_fingerprint
    assert first.route_fingerprint != check(a, st, gamma_limit=0.99).route_fingerprint
    st[1].representation.value = 0.03
    assert check(a, st).findings[0].code == "REPRESENTATION_MISMATCH"
    with pytest.raises(ValueError):
        certify_route([stage(a), stage(a, number=1)], signed_prediction=a, verification_secret=SECRET, beta={"gamma_max": float("nan")})


def test_forged_signature_is_rejected_even_when_every_stage_matches():
    a = receipt()
    modified = replace(a, score=0.9)
    l, r = stage(modified), stage(modified, number=1)
    assert check(modified, [l, r]).status == RouteStatus.REJECTED
    assert verify_prediction(modified, secret=SECRET) == RouteStatus.REJECTED


def test_missing_signer_or_missing_contract_abstains():
    a = receipt()
    steps = [stage(a), stage(a, number=1)]
    assert certify_route(steps, signed_prediction=a, verification_secret=None, beta=BETA).status == RouteStatus.INSUFFICIENT
    assert check(replace(a, signature=""), steps).status == RouteStatus.INSUFFICIENT
    steps[1].metadata.pop("semantic_contract")
    assert check(a, steps).status == RouteStatus.INSUFFICIENT


def test_missing_trace_fails_closed():
    a = receipt()
    steps = [stage(a), stage(a, number=1)]
    steps[1].trace = replace(steps[1].trace, source="")
    assert check(a, steps).status == RouteStatus.INSUFFICIENT


def test_rule_collision_is_rejected_even_with_correct_source_reference():
    a = receipt()
    l, r = stage(a), stage(a, number=1)
    r.rules = [Rule("rule-0", {"risk": "high"}, "approve")]
    assert check(a, [l, r]).status == RouteStatus.REJECTED


def test_missing_route_or_invalid_policy_parameters():
    a = receipt()
    s = stage(a)
    assert check(a, [s]).status == RouteStatus.INSUFFICIENT
    with pytest.raises(ValueError):
        check(a, [s, stage(a, number=1)], score_tolerance=float("nan"))
    with pytest.raises(ValueError):
        check(a, [s, stage(a, number=1)], gamma_limit=1.2)


def test_source_reference_does_not_accidentally_accept_invalid_shape():
    a = receipt()
    for invalid in [replace(a, source_sha256="not-a-hash"), replace(a, model_sha256=None)]:
        assert verify_prediction(invalid, secret=SECRET) == RouteStatus.INSUFFICIENT


def test_full_route_with_independently_measured_additive_effects():
    # Current repo's real SDK module is used in CI. Older research ZIPs do not
    # include the verifier, so their local copy runs the semantic suite only.
    from fuzzyxai.selective_observer import local_effects as verifier
    from fuzzyxai.selective_observer import evidence_identity as identities
    from fuzzyxai.core.certified_route import certify_route_with_local_effects

    a = receipt()
    source = hashlib.sha256(b"local attributions").hexdigest()
    reference = hashlib.sha256(b"reference vector").hexdigest()
    local = identities.LocalExplanationEvidence(
        object_id=a.object_id, model_version=a.model_version,
        target_class=a.event_id, scope="local",
        feature_names=("a", "b"), contributions=(0.2, 0.3),
        source_sha256=source, input_sha256=a.source_sha256,
        reference_sha256=reference,
        attribution_protocol=verifier.SUPPORTED_PROTOCOL,
        score_channel="target_logit",
    )
    effects = verifier.LocalEffectEvidence(
        protocol=verifier.SUPPORTED_PROTOCOL, object_id=a.object_id,
        model_version=a.model_version, target_class=a.event_id,
        feature_names=("a", "b"), effect_on_score=(0.2, 0.3),
        reference_score=0.6, explained_score=1.1,
        input_sha256=a.source_sha256, reference_sha256=reference,
    )
    route = [stage(a), stage(a, number=1)]

    def verify(l, e):
        return certify_route_with_local_effects(
            route, signed_prediction=a, verification_secret=SECRET,
            beta=BETA, local_evidence=l, source_model_effects=e,
        )

    assert verify(local, effects).status == RouteStatus.CERTIFIED
    # Zero-sum mutation survives a global reconstruction-only check.
    bad = replace(local, contributions=(0.3, 0.2))
    assert verify(bad, effects).status == RouteStatus.REJECTED
    assert verify(replace(local, reference_sha256="f" * 64), effects).status == RouteStatus.REJECTED
    assert verify(replace(local, input_sha256="f" * 64), effects).status == RouteStatus.REJECTED
    assert verify(None, effects).status == RouteStatus.INSUFFICIENT
    assert verify(replace(local, attribution_protocol="nonlinear_shap"), effects).status == RouteStatus.INSUFFICIENT


def test_json_cli_round_trip_with_externally_signed_witness(tmp_path, monkeypatch):
    import json
    from dataclasses import asdict
    from fuzzyxai.core.certified_route import certify_json_document, main

    a = receipt()
    route = [stage(a), stage(a, number=1)]
    def as_record(s):
        return {
            "terms": sorted(s.terms), "representation": {"type": "scalar", "value": s.metadata["risk"]},
            "rules": [{"name": r.name, "conditions": dict(r.conditions), "conclusion": r.conclusion} for r in s.rules],
            "activations": s.activations, "uncertainty": s.uncertainty,
            "trace": s.trace.as_dict(), "metadata": s.metadata,
        }
    doc = {"schema": "fuzzyxai-certified-route-v1", "signed_prediction": asdict(a),
           "stages": [as_record(s) for s in route], "policy": {"beta": BETA}}
    assert certify_json_document(doc, verification_secret=SECRET).status == RouteStatus.CERTIFIED
    inp, out = tmp_path / "route.json", tmp_path / "certificate.json"
    inp.write_text(json.dumps(doc), encoding="utf8")
    monkeypatch.setenv("FUZZYXAI_REFERENCE_KEY", SECRET.decode())
    assert main(["--input", str(inp), "--output", str(out)]) == 0
    assert json.loads(out.read_text())["status"] == "certified"
    monkeypatch.delenv("FUZZYXAI_REFERENCE_KEY")
    assert main(["--input", str(inp), "--output", str(out)]) == 3
    assert json.loads(out.read_text())["status"] == "insufficient_evidence"
    doc["stages"][-1]["representation"] = {"type": "unhandled", "value": 0.7}
    assert certify_json_document(doc, verification_secret=SECRET).status == RouteStatus.INSUFFICIENT


def test_branching_dag_uses_all_paths_and_requires_terminal_unanimity():
    from fuzzyxai.core.certified_route import certify_branching_route
    a = receipt()
    nodes = {k: stage(a, number=i) for i, k in enumerate(("root", "left", "right", "merge"))}
    edges = [("root", "left"), ("root", "right"), ("left", "merge"), ("right", "merge")]
    params = dict(nodes=nodes, edges=edges, root_id="root", terminal_ids=["merge"],
                  signed_prediction=a, verification_secret=SECRET, beta=BETA)
    assert certify_branching_route(**params).status == RouteStatus.CERTIFIED
    nodes["right"].metadata["risk"] = 0.33
    result = certify_branching_route(**params)
    assert result.status == RouteStatus.REJECTED
    assert result.action == "block"


def test_branching_dag_rejects_cycles_abstains_on_incomplete_sinks():
    from fuzzyxai.core.certified_route import certify_branching_route
    a = receipt()
    nodes = {k: stage(a, number=i) for i, k in enumerate(("root", "left", "right"))}
    kwargs = dict(nodes=nodes, root_id="root", signed_prediction=a,
                  verification_secret=SECRET, beta=BETA)
    assert certify_branching_route(edges=[("root", "left"), ("root", "right")],
                                  terminal_ids=["left", "right"], **kwargs).status == RouteStatus.CERTIFIED
    assert certify_branching_route(edges=[("root", "left"), ("root", "right")],
                                  terminal_ids=["left"], **kwargs).status == RouteStatus.INSUFFICIENT
    # A cycle with a separate terminal must be rejected structurally.
    assert certify_branching_route(edges=[("root", "left"), ("left", "root"),
                                        ("root", "right")], terminal_ids=["right"],
                                  **kwargs).status in (RouteStatus.REJECTED, RouteStatus.INSUFFICIENT)


def test_branching_dag_path_explosion_abstains_not_implicitly_accepts():
    from fuzzyxai.core.certified_route import certify_branching_route
    a = receipt()
    nodes = {k: stage(a, number=i) for i, k in enumerate(("root", "a", "b", "x", "y"))}
    edges = [("root", "a"), ("root", "b"), ("a", "x"), ("a", "y"),
             ("b", "x"), ("b", "y")]
    result = certify_branching_route(nodes, edges, root_id="root", terminal_ids=["x", "y"],
                                     signed_prediction=a, verification_secret=SECRET,
                                     beta=BETA, max_paths=3)
    assert result.status == RouteStatus.INSUFFICIENT
    assert result.findings[0].code == "GRAPH_PATH_LIMIT"

def test_real_sklearn_model_sdk_response_binding():
    """Exercise the actual upstream score-response collector on a fitted model."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from fuzzyxai.core.certified_route import certify_route_with_local_effects
    from fuzzyxai.selective_observer.evidence_identity import LocalExplanationEvidence
    from fuzzyxai.selective_observer.local_effects import (
        SUPPORTED_PROTOCOL, collect_linear_reference_effects_batched,
        hash_numeric_input,
    )

    X = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1], [1, 2]], dtype=float)
    y = np.asarray([0, 0, 0, 1, 1, 1], dtype=int)
    model = LogisticRegression(random_state=17, max_iter=500).fit(X, y)
    sample = tuple(float(x) for x in X[-2])
    reference = (0.0, 0.0)
    axis = ("f0", "f1")
    input_hash = hash_numeric_input(axis, sample)
    ref_hash = hash_numeric_input(axis, reference)
    model_hash = hashlib.sha256(model.coef_.tobytes() + model.intercept_.tobytes()).hexdigest()
    a = sign_prediction(SignedPrediction(
        object_id="independent:test-4", event_id="1",
        model_version="logreg-fitted-v1", unit="probability",
        score=float(model.predict_proba(np.asarray([sample]))[0, 1]),
        source_sha256=input_hash, model_sha256=model_hash,
        trace_checksum=hashlib.sha256(b"model-prediction").hexdigest(),
    ), secret=SECRET)
    stages = [stage(a), stage(a, number=1)]
    effects = collect_linear_reference_effects_batched(
        object_id=a.object_id, model_version=a.model_version,
        target_class=a.event_id, feature_names=axis,
        input_values=sample, reference_values=reference,
        target_logits=lambda rows: tuple(
            float(v) for v in model.decision_function(np.asarray(rows, dtype=float))
        ),
    )
    contributions = tuple(float(c * x) for c, x in zip(model.coef_[0], sample))
    local = LocalExplanationEvidence(
        object_id=a.object_id, model_version=a.model_version,
        target_class=a.event_id, scope="local", feature_names=axis,
        contributions=contributions,
        source_sha256=hashlib.sha256(b"attribution").hexdigest(),
        input_sha256=input_hash, reference_sha256=ref_hash,
        attribution_protocol=SUPPORTED_PROTOCOL, score_channel="target_logit",
    )

    def check_effect(record):
        return certify_route_with_local_effects(
            stages, signed_prediction=a, verification_secret=SECRET,
            beta=BETA, local_evidence=record, source_model_effects=effects,
        )

    good = check_effect(local)
    assert good.status is RouteStatus.CERTIFIED, good.to_dict()
    assert good.scope == "signed_source_semantic_route_and_independent_linear_response_v1"
    forged = replace(local, contributions=(contributions[0] + 0.1, contributions[1] - 0.1))
    assert check_effect(forged).status is RouteStatus.REJECTED
    assert check_effect(replace(local, reference_sha256="f" * 64)).status is RouteStatus.REJECTED
    assert check_effect(replace(local, input_sha256="f" * 64)).status is RouteStatus.REJECTED
