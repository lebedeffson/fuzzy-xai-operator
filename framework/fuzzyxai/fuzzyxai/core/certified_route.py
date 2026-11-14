"""Fail-closed, provenance-bound composition of a *linear* explanation route.

This is an opt-in contract beside the existing ``compose`` function.  It
certifies consistency of **declared** explanations with an independent signed
prediction record; it cannot certify clinical correctness, causal meaning, or
a compromised signer.  Transitions v1 cover identity and binary complement.
"""

from __future__ import annotations

import hmac
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from enum import Enum
from hashlib import sha256
from typing import Any

from .composition import compose
from .diagnostics import DiagnosticState
from .explanation_object import ExplanationObject

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL = (",", ":")
_REQUIRED_CONTRACT = (
    "object_id", "event_id", "model_version", "unit", "source_sha256",
    "model_sha256",
)


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=_CANONICAL, ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


class RouteStatus(str, Enum):
    CERTIFIED = "certified"
    REJECTED = "rejected"
    INSUFFICIENT = "insufficient_evidence"


@dataclass(frozen=True)
class SignedPrediction:
    """Receipt signed by an *external trusted model/source adapter*.

    The same process that evaluates the untrusted explanation must not create
    its own anchor from that explanation.  HMAC is authentication only when the
    independent producer protects the secret.  SHA-256 values identify bytes
    supplied by the producer; this class cannot independently read those bytes.
    """

    object_id: str
    event_id: str
    model_version: str
    unit: str
    score: float
    source_sha256: str
    model_sha256: str
    trace_checksum: str
    signature: str = ""
    # Producer-authorized complementary class is part of the authenticated record.
    complement_event_id: str | None = None

    def payload(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "signature"}

    def valid_shape(self) -> bool:
        keys = ("object_id", "event_id", "model_version", "unit", "trace_checksum")
        if not all(isinstance(getattr(self, k), str) and getattr(self, k).strip() for k in keys):
            return False
        if self.unit != "probability":
            return False
        if self.complement_event_id is not None and (
            not isinstance(self.complement_event_id, str)
            or not self.complement_event_id.strip()
            or self.complement_event_id == self.event_id
        ):
            return False
        if not isinstance(self.score, (float, int)) or isinstance(self.score, bool):
            return False
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            return False
        return all(isinstance(value, str) and bool(_HEX64.fullmatch(value))
                   for value in (self.source_sha256, self.model_sha256))


def sign_prediction(record: SignedPrediction, *, secret: bytes) -> SignedPrediction:
    """Execute only inside the *trusted producer*, never in the observer."""
    if not record.valid_shape():
        raise ValueError("invalid prediction snapshot")
    if not isinstance(secret, bytes) or len(secret) < 16:
        raise ValueError("HMAC key must be bytes with at least 16 bytes")
    tag = hmac.new(secret, _canonical(record.payload()), sha256).hexdigest()
    return replace(record, signature=tag)


def verify_prediction(record: SignedPrediction | None, *, secret: bytes | None) -> RouteStatus:
    if record is None or not isinstance(record, SignedPrediction) or not record.valid_shape():
        return RouteStatus.INSUFFICIENT
    if not isinstance(secret, bytes) or len(secret) < 16:
        return RouteStatus.INSUFFICIENT
    if not isinstance(record.signature, str) or not _HEX64.fullmatch(record.signature):
        return RouteStatus.INSUFFICIENT if not record.signature else RouteStatus.REJECTED
    digest = hmac.new(secret, _canonical(record.payload()), sha256).hexdigest()
    return RouteStatus.CERTIFIED if hmac.compare_digest(digest, record.signature) else RouteStatus.REJECTED


_EFFECT_WITNESS_FIELDS = (
    "protocol", "object_id", "model_version", "target_class", "feature_names",
    "effect_on_score", "reference_score", "explained_score",
    "input_sha256", "reference_sha256",
)


def _local_effect_digest(model_effects: Any) -> str:
    """Digest the exact SDK response. A digest is not evidence of actual execution."""
    payload = {field: getattr(model_effects, field) for field in _EFFECT_WITNESS_FIELDS}
    return sha256(_canonical(payload)).hexdigest()


@dataclass(frozen=True)
class SignedLocalEffectWitness:
    """HMAC receipt for a separately attested model-response record."""

    source_signature: str
    effects_sha256: str
    producer_id: str
    signature: str = ""

    def payload(self) -> dict[str, str]:
        return {
            "source_signature": self.source_signature,
            "effects_sha256": self.effects_sha256,
            "producer_id": self.producer_id,
        }

    def valid_shape(self) -> bool:
        return (
            isinstance(self.source_signature, str)
            and bool(_HEX64.fullmatch(self.source_signature))
            and isinstance(self.effects_sha256, str)
            and bool(_HEX64.fullmatch(self.effects_sha256))
            and isinstance(self.producer_id, str)
            and bool(self.producer_id.strip())
        )


def sign_local_effect_witness(
    source_model_effects: Any, *, source_signature: str,
    producer_id: str, secret: bytes,
) -> SignedLocalEffectWitness:
    """Call only inside a trusted independent model-response producer."""
    if not isinstance(secret, bytes) or len(secret) < 16:
        raise ValueError("HMAC key must be bytes with at least 16 bytes")
    try:
        digest = _local_effect_digest(source_model_effects)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("model-response evidence is not canonically serializable") from exc
    receipt = SignedLocalEffectWitness(
        source_signature=source_signature, effects_sha256=digest,
        producer_id=producer_id,
    )
    if not receipt.valid_shape():
        raise ValueError("invalid local model-effect witness")
    tag = hmac.new(secret, _canonical(receipt.payload()), sha256).hexdigest()
    return replace(receipt, signature=tag)


def verify_local_effect_witness(
    receipt: SignedLocalEffectWitness | None, *,
    source_model_effects: Any, source_signature: str,
    secret: bytes | None, expected_producer_id: str | None = None,
) -> RouteStatus:
    """Authenticate the probe record and bind it to the signed prediction."""
    if receipt is None or not isinstance(receipt, SignedLocalEffectWitness) or not receipt.valid_shape():
        return RouteStatus.INSUFFICIENT
    if not isinstance(secret, bytes) or len(secret) < 16:
        return RouteStatus.INSUFFICIENT
    if not isinstance(receipt.signature, str) or not _HEX64.fullmatch(receipt.signature):
        return RouteStatus.INSUFFICIENT if not receipt.signature else RouteStatus.REJECTED
    if receipt.source_signature != source_signature:
        return RouteStatus.REJECTED
    expected = hmac.new(secret, _canonical(receipt.payload()), sha256).hexdigest()
    if not hmac.compare_digest(expected, receipt.signature):
        return RouteStatus.REJECTED
    if expected_producer_id is not None:
        if not isinstance(expected_producer_id, str) or not expected_producer_id.strip():
            return RouteStatus.INSUFFICIENT
        if receipt.producer_id != expected_producer_id:
            return RouteStatus.REJECTED
    try:
        response_digest = _local_effect_digest(source_model_effects)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return RouteStatus.INSUFFICIENT
    return (RouteStatus.CERTIFIED if hmac.compare_digest(receipt.effects_sha256, response_digest)
            else RouteStatus.REJECTED)


@dataclass(frozen=True)
class RouteFinding:
    stage: int
    code: str
    reason: str
    status: RouteStatus


@dataclass(frozen=True)
class RouteCertificate:
    status: RouteStatus
    action: str
    route_fingerprint: str | None
    reference_signature: str | None
    edge_gamma: tuple[float, ...]
    findings: tuple[RouteFinding, ...]
    # No claim of authentic model bytes, XAI faithfulness or predictive validity.
    scope: str = "declared_semantic_route_and_signed_source_reference_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value, "action": self.action,
            "route_fingerprint": self.route_fingerprint,
            "reference_signature": self.reference_signature,
            "edge_gamma": list(self.edge_gamma),
            "findings": [
                {"stage": f.stage, "code": f.code, "reason": f.reason,
                 "status": f.status.value} for f in self.findings
            ],
            "scope": self.scope,
        }


def _issue(
    status: RouteStatus, index: int, code: str, reason: str,
    anchor: SignedPrediction | None, gammas: Sequence[float] = (),
) -> RouteCertificate:
    action = "block" if status == RouteStatus.REJECTED else "review_required"
    return RouteCertificate(
        status=status, action=action, route_fingerprint=None,
        reference_signature=(anchor.signature if anchor else None),
        edge_gamma=tuple(gammas),
        findings=(RouteFinding(index, code, reason, status),),
    )


def _finite_probability(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    number = float(value)
    return number if math.isfinite(number) and 0 <= number <= 1 else None


def _rule_trace(item: ExplanationObject) -> dict[str, Any]:
    return {
        "terms": sorted(item.terms),
        "rules": sorted(r.signature() for r in item.rules),
        "activations": dict(sorted(item.activations.items())),
        "uncertainty": float(item.uncertainty),
        "reduction_loss": float(item.reduction_loss),
        "semantic_contract": item.metadata.get("semantic_contract"),
        "risk": item.metadata.get("risk"),
        "semantic_transition": item.metadata.get("semantic_transition"),
        "trace": item.trace.as_dict(),
    }


def certify_route(
    stages: Sequence[ExplanationObject],
    *,
    signed_prediction: SignedPrediction | None,
    verification_secret: bytes | None,
    beta: Mapping[str, float],
    score_tolerance: float = 1e-8,
    gamma_limit: float = 1.0,
) -> RouteCertificate:
    """Audit sequential composition without mutating the source model or stages.

    Certifying the *same event* permits only score identity; switching between
    two complementary events requires an explicit, correctly oriented
    ``semantic_transition``. Arbitrary calibration/normalization is unsupported
    and is not silently interpreted as a valid identity transition.
    """
    if isinstance(score_tolerance, bool) or not isinstance(score_tolerance, (float, int)) or not math.isfinite(score_tolerance) or score_tolerance < 0:
        raise ValueError("score_tolerance must be finite and nonnegative")
    if isinstance(gamma_limit, bool) or not isinstance(gamma_limit, (float, int)) or not math.isfinite(gamma_limit) or not 0 <= gamma_limit <= 1:
        raise ValueError("gamma_limit must be in [0,1]")
    if not isinstance(beta, Mapping) or any(
        not isinstance(k, str) or not k.strip()
        or not isinstance(v, (float, int)) or isinstance(v, bool)
        or not math.isfinite(v) or v < 0
        for k, v in beta.items()
    ):
        raise ValueError("beta weights must be finite and nonnegative")
    if "gamma_max" in beta and not 0.0 <= beta["gamma_max"] <= 1.0:
        raise ValueError("beta gamma_max must be in [0,1]")
    if len(stages) < 2:
        return _issue(RouteStatus.INSUFFICIENT, -1, "ROUTE_TOO_SHORT", "at least two stages required", signed_prediction)
    signed_status = verify_prediction(signed_prediction, secret=verification_secret)
    if signed_status != RouteStatus.CERTIFIED:
        return _issue(signed_status, -1, "SOURCE_REFERENCE_UNVERIFIED", "trusted prediction receipt absent or invalid", signed_prediction)
    assert signed_prediction is not None

    prev_event = signed_prediction.event_id
    gammas: list[float] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, ExplanationObject):
            return _issue(RouteStatus.INSUFFICIENT, index, "STAGE_MISSING", "typed explanation missing", signed_prediction, gammas)
        contract = stage.metadata.get("semantic_contract")
        if not isinstance(contract, Mapping) or any(
            not isinstance(contract.get(k), str) or not contract[k].strip() for k in _REQUIRED_CONTRACT
        ):
            return _issue(RouteStatus.INSUFFICIENT, index, "CONTRACT_MISSING", "mandatory semantic fields absent", signed_prediction, gammas)
        for field in ("object_id", "model_version", "unit", "source_sha256", "model_sha256"):
            if contract[field] != getattr(signed_prediction, field):
                return _issue(RouteStatus.REJECTED, index, "REFERENCE_MISMATCH", f"{field} differs from signed source", signed_prediction, gammas)
        if not stage.trace.source or not stage.trace.checksum:
            return _issue(RouteStatus.INSUFFICIENT, index, "TRACE_MISSING", "stage source/checksum missing", signed_prediction, gammas)
        if index == 0 and stage.trace.checksum != signed_prediction.trace_checksum:
            return _issue(RouteStatus.REJECTED, index, "TRACE_MISMATCH", "source trace differs from receipt", signed_prediction, gammas)
        p = _finite_probability(stage.metadata.get("risk"))
        if p is None:
            return _issue(RouteStatus.INSUFFICIENT, index, "SCORE_MISSING", "finite probability unavailable", signed_prediction, gammas)
        # A scalar explanation must display the same value as the checked risk.
        represented = getattr(stage.representation, "value", None)
        if represented is not None:
            represented_value = _finite_probability(represented)
            if represented_value is None:
                return _issue(RouteStatus.INSUFFICIENT, index, "REPRESENTATION_UNKNOWN", "invalid scalar representation", signed_prediction, gammas)
            if abs(represented_value - p) > score_tolerance:
                return _issue(RouteStatus.REJECTED, index, "REPRESENTATION_MISMATCH", "scalar explanation disagrees with declared score", signed_prediction, gammas)
        event = contract["event_id"]
        transition = stage.metadata.get("semantic_transition")
        if event == prev_event:
            if transition is not None:
                return _issue(RouteStatus.REJECTED, index, "UNEXPECTED_TRANSITION", "event unchanged but transition declared", signed_prediction, gammas)
            expected = (signed_prediction.score if event == signed_prediction.event_id
                        else 1.0 - signed_prediction.score)
        else:
            if index == 0 or not isinstance(transition, Mapping) or (
                transition.get("kind") != "complement" or
                transition.get("from_event") != prev_event or
                transition.get("to_event") != event
            ):
                return _issue(RouteStatus.REJECTED, index, "EVENT_MISMATCH", "unregistered or misoriented event conversion", signed_prediction, gammas)
            # Untrusted explanation metadata cannot authorize an arbitrary complement.
            complement = signed_prediction.complement_event_id
            if complement is None or {prev_event, event} != {signed_prediction.event_id, complement}:
                return _issue(RouteStatus.REJECTED, index, "COMPLEMENT_NOT_AUTHORIZED", "binary complement pair absent from signed origin", signed_prediction, gammas)
            expected = (signed_prediction.score if event == signed_prediction.event_id
                        else 1.0 - signed_prediction.score)
        if abs(p - expected) > score_tolerance:
            return _issue(RouteStatus.REJECTED, index, "SCORE_MISMATCH", "score disagrees with signed source and declared mapping", signed_prediction, gammas)
        prev_event = event
        if index:
            pair = compose(stages[index - 1], stage, beta)
            if isinstance(pair, DiagnosticState):
                status = RouteStatus.REJECTED if pair.is_critical() else RouteStatus.INSUFFICIENT
                return _issue(status, index, "COMPOSITION_UNAVAILABLE", pair.reason, signed_prediction, gammas)
            gamma = _finite_probability(pair.metadata.get("gamma"))
            if gamma is None:
                return _issue(RouteStatus.INSUFFICIENT, index, "GAMMA_UNKNOWN", "composition disagreement missing", signed_prediction, gammas)
            if gamma > gamma_limit:
                return _issue(RouteStatus.REJECTED, index, "GAMMA_EXCEEDED", "declared disagreement exceeds policy", signed_prediction, gammas)
            gammas.append(gamma)
    trace = {
        "version": 2, "source_signature": signed_prediction.signature,
        "verification_policy": {
            "beta": dict(sorted(beta.items())),
            "score_tolerance": float(score_tolerance),
            "gamma_limit": float(gamma_limit),
        },
        "stage_claims": [_rule_trace(stage) for stage in stages],
        "edge_gamma": gammas,
    }
    try:
        fingerprint = sha256(_canonical(trace)).hexdigest()
    except (ValueError, TypeError):
        return _issue(RouteStatus.INSUFFICIENT, -1, "FINGERPRINT_FAILED", "unserializable route evidence", signed_prediction, gammas)
    return RouteCertificate(
        RouteStatus.CERTIFIED, "eligible_for_policy_review", fingerprint,
        signed_prediction.signature, tuple(gammas), (),
    )


def certify_route_with_local_effects(
    stages: Sequence[ExplanationObject],
    *,
    signed_prediction: SignedPrediction | None,
    verification_secret: bytes | None,
    beta: Mapping[str, float],
    local_evidence: Any,
    source_model_effects: Any,
    absolute_tolerance: float = 1e-6,
    score_tolerance: float = 1e-8,
    gamma_limit: float = 1.0,
) -> RouteCertificate:
    """Stronger certificate using the SDK's independent *linear* logit audit.

    The source-model effects must be supplied by a separately trusted adapter.
    The observer never trains or executes the source model itself. Non-linear
    SHAP/LIME/interaction attributions are deliberately NOT certified here.
    """
    if (isinstance(absolute_tolerance, bool)
            or not isinstance(absolute_tolerance, (float, int))
            or not math.isfinite(absolute_tolerance) or absolute_tolerance < 0):
        raise ValueError("absolute_tolerance must be finite and nonnegative")
    result = certify_route(
        stages, signed_prediction=signed_prediction,
        verification_secret=verification_secret, beta=beta,
        score_tolerance=score_tolerance, gamma_limit=gamma_limit,
    )
    if result.status is not RouteStatus.CERTIFIED:
        return result
    assert signed_prediction is not None
    try:
        from fuzzyxai.selective_observer.evidence_identity import LocalEvidenceStatus
        from fuzzyxai.selective_observer.local_effects import (
            SUPPORTED_PROTOCOL,
            verify_additive_local_effects,
        )
    except ImportError:
        return _issue(RouteStatus.INSUFFICIENT, -1, "RESPONSE_VERIFIER_UNAVAILABLE",
                      "linear-additive verifier not installed", signed_prediction, result.edge_gamma)
    if local_evidence is None or source_model_effects is None:
        return _issue(RouteStatus.INSUFFICIENT, -1, "RESPONSE_EVIDENCE_MISSING",
                      "both attribution and independent model effects are required",
                      signed_prediction, result.edge_gamma)
    if getattr(local_evidence, "attribution_protocol", None) != SUPPORTED_PROTOCOL:
        return _issue(RouteStatus.INSUFFICIENT, -1, "UNSUPPORTED_ATTRIBUTION_PROTOCOL",
                      "only linear-reference additive target logits supported",
                      signed_prediction, result.edge_gamma)
    if getattr(local_evidence, "score_channel", None) != "target_logit":
        return _issue(RouteStatus.INSUFFICIENT, -1, "UNSUPPORTED_SCORE_CHANNEL",
                      "target-logit score channel must be declared", signed_prediction, result.edge_gamma)
    for key, expected in (("object_id", signed_prediction.object_id),
                          ("model_version", signed_prediction.model_version),
                          ("target_class", signed_prediction.event_id),
                          ("scope", "local"),
                          ("input_sha256", signed_prediction.source_sha256)):
        actual = getattr(local_evidence, key, None)
        if actual is None or actual == "":
            return _issue(RouteStatus.INSUFFICIENT, 0, "LOCAL_BINDING_MISSING",
                          f"local {key} absent", signed_prediction, result.edge_gamma)
        if actual != expected:
            return _issue(RouteStatus.REJECTED, 0, "LOCAL_BINDING_MISMATCH",
                          f"local {key} differs from signed source", signed_prediction, result.edge_gamma)
    if getattr(source_model_effects, "input_sha256", None) != signed_prediction.source_sha256:
        return _issue(RouteStatus.REJECTED, 0, "PROBE_INPUT_MISMATCH",
                      "model response probe did not use signed input", signed_prediction, result.edge_gamma)
    ref_local = getattr(local_evidence, "reference_sha256", None)
    ref_probe = getattr(source_model_effects, "reference_sha256", None)
    if not ref_local or not ref_probe:
        return _issue(RouteStatus.INSUFFICIENT, 0, "REFERENCE_MISSING",
                      "reference identities not supplied", signed_prediction, result.edge_gamma)
    if ref_local != ref_probe:
        return _issue(RouteStatus.REJECTED, 0, "REFERENCE_MISMATCH",
                      "model and explainer used different reference vectors", signed_prediction, result.edge_gamma)
    observed = verify_additive_local_effects(
        local_evidence, source_model_effects, absolute_tolerance=absolute_tolerance,
    )
    if observed.status is LocalEvidenceStatus.INCOMPATIBLE:
        return _issue(RouteStatus.REJECTED, 0, "LOCAL_EFFECT_MISMATCH",
                      ", ".join(observed.contradictions), signed_prediction, result.edge_gamma)
    if observed.status is not LocalEvidenceStatus.CERTIFIED:
        return _issue(RouteStatus.INSUFFICIENT, 0, "LOCAL_EFFECT_UNKNOWN",
                      ", ".join(observed.missing_fields), signed_prediction, result.edge_gamma)
    evidence_record = {
        "route": result.route_fingerprint,
        "protocol": SUPPORTED_PROTOCOL,
        "source_effects": {
            "object_id": source_model_effects.object_id,
            "model_version": source_model_effects.model_version,
            "target_class": source_model_effects.target_class,
            "feature_names": source_model_effects.feature_names,
            "effect_on_score": source_model_effects.effect_on_score,
            "reference_score": source_model_effects.reference_score,
            "explained_score": source_model_effects.explained_score,
            "input_sha256": source_model_effects.input_sha256,
            "reference_sha256": source_model_effects.reference_sha256,
        },
        "declared_contributions": local_evidence.contributions,
        "absolute_tolerance": float(absolute_tolerance),
    }
    try:
        fingerprint = sha256(_canonical(evidence_record)).hexdigest()
    except (ValueError, TypeError):
        return _issue(RouteStatus.INSUFFICIENT, -1, "EFFECT_FINGERPRINT_FAILED",
                      "response evidence not serializable", signed_prediction, result.edge_gamma)
    return replace(
        result, route_fingerprint=fingerprint,
        scope="signed_source_semantic_route_and_declared_linear_response_v1",
    )


def certify_route_with_attested_local_effects(
    stages: Sequence[ExplanationObject], *,
    signed_prediction: SignedPrediction | None,
    verification_secret: bytes | None,
    beta: Mapping[str, float],
    local_evidence: Any,
    source_model_effects: Any,
    signed_effect_witness: SignedLocalEffectWitness | None,
    effects_verification_secret: bytes | None,
    expected_effect_producer_id: str | None = None,
    absolute_tolerance: float = 1e-6,
    score_tolerance: float = 1e-8,
    gamma_limit: float = 1.0,
) -> RouteCertificate:
    """Authenticate a separate model-response receipt before certification.

    Successful verification attests reported linear response evidence, not
    causal faithfulness or actual execution by a compromised producer.
    """
    base = certify_route_with_local_effects(
        stages, signed_prediction=signed_prediction,
        verification_secret=verification_secret, beta=beta,
        local_evidence=local_evidence, source_model_effects=source_model_effects,
        absolute_tolerance=absolute_tolerance,
        score_tolerance=score_tolerance, gamma_limit=gamma_limit,
    )
    if base.status is not RouteStatus.CERTIFIED:
        return base
    assert signed_prediction is not None
    if (isinstance(verification_secret, bytes)
            and isinstance(effects_verification_secret, bytes)
            and hmac.compare_digest(verification_secret, effects_verification_secret)):
        return _issue(RouteStatus.INSUFFICIENT, -1, "EFFECT_SIGNER_KEY_REUSED",
                      "independent model-response signer must have a distinct key",
                      signed_prediction, base.edge_gamma)
    witnessed = verify_local_effect_witness(
        signed_effect_witness, source_model_effects=source_model_effects,
        source_signature=signed_prediction.signature,
        secret=effects_verification_secret,
        expected_producer_id=expected_effect_producer_id,
    )
    if witnessed is not RouteStatus.CERTIFIED:
        return _issue(
            witnessed, -1, "EFFECT_WITNESS_UNVERIFIED",
            "separate model-response receipt absent or invalid",
            signed_prediction, base.edge_gamma,
        )
    assert signed_effect_witness is not None
    fingerprint = sha256(_canonical({
        "contract": "dual_attested_linear_response_v1",
        "base_route": base.route_fingerprint,
        "effect_witness": signed_effect_witness.payload(),
        "effect_signature": signed_effect_witness.signature,
    })).hexdigest()
    return replace(
        base, route_fingerprint=fingerprint,
        scope="signed_prediction_and_independently_attested_linear_response_v1",
    )


class _ScalarRepresentation:
    def __init__(self, value: float) -> None:
        self.value = value

    def distance(self, other: Any) -> float:
        if not isinstance(other, _ScalarRepresentation):
            raise TypeError("incompatible representations")
        return abs(self.value - other.value)


def certify_json_document(document: Mapping[str, Any], *, verification_secret: bytes | None) -> RouteCertificate:
    """Portable, strictly typed JSON entrypoint for offline route verification.

    Only scalar representations are supported by this JSON v1 contract. Other
    representations must be normalized by a source adapter and passed to the
    Python SDK; they are not silently stringified or coerced in the CLI.
    """
    from .explanation_object import Rule, Trace

    if not isinstance(document, Mapping) or document.get("schema") != "fuzzyxai-certified-route-v1":
        return _issue(RouteStatus.INSUFFICIENT, -1, "SCHEMA_UNKNOWN", "unsupported or absent schema", None)
    source = document.get("signed_prediction")
    try:
        anchor = SignedPrediction(**source) if isinstance(source, dict) else None
    except (TypeError, ValueError):
        anchor = None
    policy = document.get("policy")
    if not isinstance(policy, dict) or not isinstance(policy.get("beta"), dict):
        return _issue(RouteStatus.INSUFFICIENT, -1, "POLICY_MISSING", "explicit policy required", anchor)
    items = document.get("stages")
    if not isinstance(items, list) or not 2 <= len(items) <= 256:
        return _issue(RouteStatus.INSUFFICIENT, -1, "STAGES_MISSING", "2..256 stages required", anchor)
    stages: list[ExplanationObject] = []
    for i, record in enumerate(items):
        try:
            if not isinstance(record, dict):
                raise TypeError("stage must be object")
            rep = record["representation"]
            if not isinstance(rep, dict) or rep.get("type") != "scalar":
                raise ValueError("only scalar representation permitted in JSON v1")
            val = _finite_probability(rep.get("value"))
            if val is None:
                raise ValueError("finite scalar in [0,1] required")
            t = record["trace"]
            trace = Trace(
                id=t["id"], version=t["version"], timestamp=t["timestamp"],
                params=t.get("params", {}), source=t["source"], checksum=t["checksum"],
            )
            rules = [Rule(name=x["name"], conditions=x["conditions"], conclusion=x["conclusion"])
                     for x in record["rules"]]
            stages.append(ExplanationObject(
                terms=set(record["terms"]), representation=_ScalarRepresentation(val),
                rules=rules, activations=dict(record["activations"]),
                uncertainty=float(record["uncertainty"]), trace=trace,
                reduction_loss=float(record.get("reduction_loss", 0.0)),
                metadata=dict(record["metadata"]),
            ))
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            return _issue(RouteStatus.INSUFFICIENT, i, "STAGE_SCHEMA_UNSUPPORTED", str(exc), anchor)
    try:
        return certify_route(
            stages, signed_prediction=anchor, verification_secret=verification_secret,
            beta=policy["beta"], score_tolerance=policy.get("score_tolerance", 1e-8),
            gamma_limit=policy.get("gamma_limit", 1.0),
        )
    except (ValueError, TypeError) as exc:
        return _issue(RouteStatus.INSUFFICIENT, -1, "POLICY_INVALID", str(exc), anchor)


def main(argv: Sequence[str] | None = None) -> int:
    """Offline JSON audit; key comes from the environment, never CLI argv."""
    import argparse
    import os
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Verify signed FuzzyXAI explanation routes")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--key-env", default="FUZZYXAI_REFERENCE_KEY")
    args = parser.parse_args(argv)
    key_value = os.environ.get(args.key_env)
    key = key_value.encode("utf-8") if key_value else None
    try:
        document = json.loads(args.input.read_text(encoding="utf-8"))
        certificate = certify_json_document(document, verification_secret=key)
        result = certificate.to_dict()
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        result = {
            "status": RouteStatus.INSUFFICIENT.value, "action": "review_required",
            "findings": [{"code": "INPUT_ERROR", "reason": str(exc)}],
        }
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        sys.stdout.write(rendered + "\n")
    return {
        RouteStatus.CERTIFIED.value: 0,
        RouteStatus.REJECTED.value: 2,
        RouteStatus.INSUFFICIENT.value: 3,
    }[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())


def certify_branching_route(
    nodes: Mapping[str, ExplanationObject],
    edges: Sequence[tuple[str, str]],
    *,
    root_id: str,
    terminal_ids: Sequence[str],
    signed_prediction: SignedPrediction | None,
    verification_secret: bytes | None,
    beta: Mapping[str, float],
    score_tolerance: float = 1e-8,
    gamma_limit: float = 1.0,
    max_paths: int = 256,
) -> RouteCertificate:
    """Conservative acyclic fork/merge verification, *not* fuzzy graph fusion.

    Every root-to-leaf path must independently certify against the SAME signed
    origin. Terminal events and scores must agree; there is no arbitrary
    aggregation across conflicting branches. Too many paths, multiple roots,
    unobserved branches, or missing conflict policy cause abstention.
    """
    if not isinstance(max_paths, int) or isinstance(max_paths, bool) or not 1 <= max_paths <= 4096:
        raise ValueError("max_paths must be integer in [1,4096]")
    if not isinstance(nodes, Mapping) or not 2 <= len(nodes) <= 256:
        return _issue(RouteStatus.INSUFFICIENT, -1, "GRAPH_NODES_MISSING", "2..256 typed nodes required", signed_prediction)
    if root_id not in nodes or not terminal_ids or any(t not in nodes for t in terminal_ids):
        return _issue(RouteStatus.INSUFFICIENT, -1, "GRAPH_ENDPOINTS_MISSING", "root/terminals invalid", signed_prediction)
    if len(set(terminal_ids)) != len(terminal_ids):
        return _issue(RouteStatus.INSUFFICIENT, -1, "GRAPH_TERMINALS_DUPLICATE", "terminal IDs must be distinct", signed_prediction)
    graph: dict[str, set[str]] = {key: set() for key in nodes}
    indegree = {key: 0 for key in nodes}
    if len(edges) > 8192:
        return _issue(RouteStatus.INSUFFICIENT, -1, "GRAPH_TOO_LARGE", "edge limit exceeded", signed_prediction)
    for edge in edges:
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            return _issue(RouteStatus.INSUFFICIENT, -1, "GRAPH_EDGE_INVALID", "edges must be pairs", signed_prediction)
        left, right = edge
        if left not in nodes or right not in nodes or left == right:
            return _issue(RouteStatus.REJECTED, -1, "GRAPH_EDGE_INVALID", "unknown endpoint or self loop", signed_prediction)
        if right in graph[left]:
            return _issue(RouteStatus.INSUFFICIENT, -1, "GRAPH_EDGE_DUPLICATE", "duplicate edge", signed_prediction)
        graph[left].add(right)
        indegree[right] += 1
    if indegree[root_id] != 0 or any(degree == 0 and key != root_id for key, degree in indegree.items()):
        return _issue(RouteStatus.INSUFFICIENT, -1, "GRAPH_ROOT_AMBIGUOUS", "multiple roots or incoming root edge", signed_prediction)
    terminals = set(terminal_ids)
    if terminals != {key for key, children in graph.items() if not children}:
        return _issue(RouteStatus.INSUFFICIENT, -1, "GRAPH_TERMINALS_INCOMPLETE", "all sinks must be registered", signed_prediction)
    # Topological check also rejects cyclic subgraphs disconnected from the root.
    pending = dict(indegree)
    ready = sorted(k for k, degree in pending.items() if degree == 0)
    seen = 0
    while ready:
        node = ready.pop(0)
        seen += 1
        for target in sorted(graph[node]):
            pending[target] -= 1
            if pending[target] == 0:
                ready.append(target)
        ready.sort()
    if seen != len(nodes):
        return _issue(RouteStatus.REJECTED, -1, "GRAPH_CYCLE", "cyclic explanation route", signed_prediction)
    paths: list[tuple[str, ...]] = []
    stack = [(root_id, (root_id,))]
    while stack:
        node, route = stack.pop()
        if node in terminals:
            paths.append(route)
            if len(paths) > max_paths:
                return _issue(RouteStatus.INSUFFICIENT, -1, "GRAPH_PATH_LIMIT", "too many root-to-sink paths", signed_prediction)
        else:
            for child in sorted(graph[node], reverse=True):
                stack.append((child, route + (child,)))
    observed_nodes = {key for path in paths for key in path}
    if observed_nodes != set(nodes):
        return _issue(RouteStatus.INSUFFICIENT, -1, "GRAPH_DISCONNECTED", "some nodes are not on a certified route", signed_prediction)
    if any(len(path) < 2 for path in paths):
        return _issue(RouteStatus.INSUFFICIENT, -1, "GRAPH_SHORT_PATH", "single-node explanation path", signed_prediction)
    signatures = []
    all_gammas: list[float] = []
    outcomes: list[tuple[str, float]] = []
    for path in paths:
        audit = certify_route(
            [nodes[key] for key in path], signed_prediction=signed_prediction,
            verification_secret=verification_secret, beta=beta,
            score_tolerance=score_tolerance, gamma_limit=gamma_limit,
        )
        if audit.status is not RouteStatus.CERTIFIED:
            issue = audit.findings[0]
            return _issue(audit.status, -1, issue.code,
                          "path " + " -> ".join(path) + ": " + issue.reason,
                          signed_prediction, all_gammas)
        signatures.append(audit.route_fingerprint)
        all_gammas.extend(audit.edge_gamma)
        final = nodes[path[-1]].metadata
        outcomes.append((final["semantic_contract"]["event_id"], final["risk"]))
    first_event, first_score = outcomes[0]
    if any(event != first_event or abs(score - first_score) > score_tolerance
           for event, score in outcomes[1:]):
        return _issue(RouteStatus.INSUFFICIENT, -1, "BRANCH_OUTCOME_CONFLICT",
                      "different terminal outcomes require an explicit fusion policy",
                      signed_prediction, all_gammas)
    digest = sha256(_canonical({
        "contract": "unanimous_branching_route_v1", "root": root_id,
        "terminals": sorted(terminals),
        "edges": sorted([list(edge) for edge in edges]),
        "path_fingerprints": signatures,
        "source_signature": signed_prediction.signature if signed_prediction else None,
    })).hexdigest()
    return RouteCertificate(
        RouteStatus.CERTIFIED, "eligible_for_policy_review", digest,
        signed_prediction.signature if signed_prediction else None,
        tuple(all_gammas), (),
        scope="all_registered_acyclic_paths_and_unanimous_terminals_v1",
    )