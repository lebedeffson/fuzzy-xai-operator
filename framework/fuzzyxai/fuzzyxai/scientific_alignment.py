"""P16: bridges runtime evidence to the dissertation's real chapter-2/3
machinery for Γ (alignment), Δ (reduction loss), and I_pre (interpretability
index) — replacing P15.7's heuristic proxies (percent-of-supporting-claims,
surrogate-fidelity-gap) with the actual tested functions:

- ``core.trust_evaluator.semantic_disagreement`` for Γ = d_E(T_ij(E_i), E_j),
- ``trust.trust_evaluator.compute_interpretability_index`` for I_pre = exp(-L(E)),
- the linear reconstruction chain already built for P15.1 for Δ_M.

Every quantity here is either a genuinely measured value or explicitly
``None``/``not_applicable`` — nothing is defaulted to 0 or fabricated when
its real inputs are absent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .core.explain_plan import ExplainPlan
from .core.explanation_object import ExplanationObject, Rule, Trace
from .core.trust_evaluator import activation_distance, jaccard_distance, representation_distance, semantic_disagreement, trace_distance
from .hierarchy.f0 import F0
from .trust.trust_evaluator import compute_interpretability_index


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class AlignmentTransform:
    """Executable, serializable T_ij between two explanation interfaces.

    A transform deliberately accepts only explicit identity mappings for
    representation, uncertainty, and trace in this first contract.  That is
    sufficient for comparable F0 channels and prevents an unmeasured coercion
    from being mistaken for alignment.
    """

    transform_id: str
    source_interface: str
    target_interface: str
    term_mapping: Mapping[str, str] = field(default_factory=dict)
    rule_mapping: Mapping[str, str] = field(default_factory=dict)
    representation_mapping: str = "identity"
    uncertainty_mapping: str = "identity"
    trace_mapping: str = "identity"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    source_refs: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AlignmentTransform:
        return cls(
            transform_id=str(data["transform_id"]),
            source_interface=str(data["source_interface"]),
            target_interface=str(data["target_interface"]),
            term_mapping={str(key): str(value) for key, value in dict(data.get("term_mapping", {})).items()},
            rule_mapping={str(key): str(value) for key, value in dict(data.get("rule_mapping", {})).items()},
            representation_mapping=str(data.get("representation_mapping", "identity")),
            uncertainty_mapping=str(data.get("uncertainty_mapping", "identity")),
            trace_mapping=str(data.get("trace_mapping", "identity")),
            parameters=dict(data.get("parameters", {})),
            source_refs=tuple(str(value) for value in data.get("source_refs", ())),
            limitations=tuple(str(value) for value in data.get("limitations", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "source_interface": self.source_interface,
            "target_interface": self.target_interface,
            "term_mapping": dict(self.term_mapping),
            "rule_mapping": dict(self.rule_mapping),
            "representation_mapping": self.representation_mapping,
            "uncertainty_mapping": self.uncertainty_mapping,
            "trace_mapping": self.trace_mapping,
            "parameters": dict(self.parameters),
            "source_refs": list(self.source_refs),
            "limitations": list(self.limitations),
        }

    def apply(self, source: ExplanationObject) -> ExplanationObject:
        """Execute T_ij or fail closed when its declared map is incomplete."""

        if source.metadata.get("interface") != self.source_interface:
            raise ValueError("source explanation does not match AlignmentTransform.source_interface")
        if self.uncertainty_mapping != "identity" or self.trace_mapping != "identity":
            raise ValueError("unsupported AlignmentTransform mapping; no unmeasured coercion is permitted")
        if any(term not in self.term_mapping for term in source.terms):
            raise ValueError("term_mapping does not cover every source term")
        if any(rule.name not in self.rule_mapping for rule in source.rules):
            raise ValueError("rule_mapping does not cover every source rule")
        mapped_rules = [Rule(self.rule_mapping[rule.name], dict(rule.conditions), rule.conclusion) for rule in source.rules]
        mapped_activations = {
            self.rule_mapping.get(name, self.term_mapping.get(name, name)): value for name, value in source.activations.items()
        }
        representation = source.representation
        transformed_memberships: dict[str, float] | None = None
        if self.representation_mapping == "risk_membership_partition":
            triangles = self.parameters.get("triangles")
            if not isinstance(representation, F0) or not isinstance(triangles, Mapping):
                raise ValueError("risk_membership_partition requires an F0 source and declared triangles")

            def partition(value: float) -> dict[str, float]:
                values: dict[str, float] = {}
                for name, raw in triangles.items():
                    if not isinstance(raw, Sequence) or len(raw) != 3:
                        raise ValueError("every membership triangle must have three points")
                    left, peak, right = (float(item) for item in raw)
                    if not left <= value <= right:
                        values[str(name)] = 0.0
                    elif (left == peak and value == left) or (peak == right and value == right) or value == peak:
                        values[str(name)] = 1.0
                    else:
                        values[str(name)] = (value - left) / (peak - left) if value < peak else (right - value) / (right - peak)
                return values

            def mapped_partition_mu(x: float, source_mu=representation.mu) -> float:
                return max(partition(_clip01(source_mu(x))).values(), default=0.0)

            transformed_memberships = partition(_clip01(representation.membership(0.0)))
            representation = F0(mapped_partition_mu, label="aligned_risk_membership_partition")
        elif self.representation_mapping == "triangular_membership":
            triangle = self.parameters.get("triangle")
            if not isinstance(representation, F0) or not isinstance(triangle, Sequence) or len(triangle) != 3:
                raise ValueError("triangular_membership requires an F0 source and a three-point triangle")
            a, b, c = (float(value) for value in triangle)

            def mapped_mu(x: float, source_mu=representation.mu, left=a, peak=b, right=c) -> float:
                value = _clip01(source_mu(x))
                if not left <= value <= right:
                    return 0.0
                if (left == peak and value == left) or (peak == right and value == right) or value == peak:
                    return 1.0
                return (value - left) / (peak - left) if value < peak else (right - value) / (right - peak)

            representation = F0(mapped_mu, label="aligned_triangular_membership")
        elif self.representation_mapping != "identity":
            raise ValueError("unsupported AlignmentTransform representation mapping")
        metadata = dict(source.metadata)
        metadata.update({"interface": self.target_interface, "alignment_transform": self.to_dict()})
        if transformed_memberships is not None:
            metadata["transformed_memberships"] = transformed_memberships
        return source.copy_with(
            terms={self.term_mapping[term] for term in source.terms},
            rules=mapped_rules,
            activations=mapped_activations,
            representation=representation,
            metadata=metadata,
        )


def build_native_explanation_object(
    fuzzy_activations: Sequence[Mapping[str, Any]],
    *,
    object_id: str,
    model_fingerprint: str,
    source: str,
    reduction_loss: float = 0.0,
) -> ExplanationObject | None:
    """The rule/fuzzy explanatory channel — only built when the model itself
    supplied real activated rules (never synthesized from contributions)."""

    terms: set[str] = set()
    rules: list[Rule] = []
    rule_activations: dict[str, float] = {}
    for raw in fuzzy_activations or ():
        rule_id = str(raw.get("rule_id", "")).strip()
        raw_terms = raw.get("terms")
        strength = raw.get("activation_strength")
        if not rule_id or not isinstance(raw_terms, Sequence) or not isinstance(strength, (int, float)):
            continue
        conditions: dict[str, str] = {}
        for term in raw_terms:
            if not isinstance(term, Mapping):
                continue
            feature = str(term.get("feature", "")).strip()
            label = str(term.get("term", "")).strip()
            if not feature or not label:
                continue
            conditions[feature] = label
            terms.add(f"{feature}:{label}")
        rules.append(Rule(rule_id, conditions, str(raw.get("conclusion", ""))))
        rule_activations[rule_id] = _clip01(strength)
    if not rules:
        return None
    top = max(rule_activations.values())
    return ExplanationObject(
        terms=terms,
        representation=F0(lambda _x, val=top: val, label="fuzzy_native"),
        rules=rules,
        activations=rule_activations,
        uncertainty=_clip01(1.0 - top),
        trace=Trace(
            id=str(object_id),
            version=model_fingerprint[:12],
            timestamp=datetime.now(UTC).isoformat(),
            source=source,
            checksum=f"{object_id}:{model_fingerprint[:12]}:{source}",
        ),
        reduction_loss=_clip01(reduction_loss),
        metadata={"channel": "fuzzy_native", "interface": "native_rules"},
    )


def build_contribution_explanation_object(
    contributions: Mapping[str, float],
    *,
    object_id: str,
    model_fingerprint: str,
    source: str,
    score: float | None,
    reduction_loss: float = 0.0,
) -> ExplanationObject | None:
    """The generic numeric local-contribution channel. Refuses to build an
    object at all when there is no real score to derive uncertainty from —
    an invented uncertainty (e.g. 0.5) would poison every downstream Γ/I_pre
    computation with a fabricated number."""

    numeric = {name: float(value) for name, value in contributions.items() if isinstance(value, (int, float))}
    if not numeric or score is None:
        return None
    max_abs = max(abs(value) for value in numeric.values()) or 1.0
    activations = {name: _clip01(abs(value) / max_abs) for name, value in numeric.items()}
    clipped_score = _clip01(score)
    return ExplanationObject(
        terms=set(numeric.keys()),
        representation=F0(lambda _x, val=clipped_score: val, label="contribution_channel"),
        rules=[],
        activations=activations,
        uncertainty=_clip01(1.0 - float(score)),
        trace=Trace(
            id=str(object_id),
            version=model_fingerprint[:12],
            timestamp=datetime.now(UTC).isoformat(),
            source=source,
            checksum=f"{object_id}:{model_fingerprint[:12]}:{source}",
        ),
        reduction_loss=_clip01(reduction_loss),
        metadata={"channel": "contribution", "interface": "contribution"},
    )


def compute_real_alignment(
    native: ExplanationObject | None,
    derived: ExplanationObject | None,
    *,
    plan: ExplainPlan,
    transform: AlignmentTransform | None,
) -> dict[str, Any]:
    """γ_ij = d_E(T_ij(E_i), E_j) via the real, tested semantic_disagreement.

    Requires two genuinely distinct explanatory channels for the same
    object (e.g. a fuzzy model's native rule activations vs. its numeric
    contribution channel). Most single-channel sklearn explanations have no
    second channel to compare against automatically — this function then
    honestly returns gamma=None, not a fabricated single-component proxy.
    """

    if native is None or derived is None:
        return {
            "gamma": None,
            "gamma_max": plan.gamma_critical,
            "certified": None,
            "status": "incomplete",
            "components": {},
            "component_status": {},
            "missing_components": ["second_explanatory_channel"],
            "weights": dict(plan.beta),
        }
    if transform is None:
        return {
            "gamma": None, "gamma_max": plan.gamma_critical, "certified": None,
            "status": "missing_required", "components": {}, "component_status": {},
            "missing_components": ["alignment_transform"], "weights": dict(plan.beta),
        }
    try:
        aligned = transform.apply(native)
    except ValueError as exc:
        return {
            "gamma": None, "gamma_max": plan.gamma_critical, "certified": None,
            "status": "missing_required", "components": {}, "component_status": {},
            "missing_components": ["alignment_transform"], "weights": dict(plan.beta),
            "reason": str(exc), "transform": transform.to_dict(),
        }
    if derived.metadata.get("interface") != transform.target_interface:
        return {
            "gamma": None, "gamma_max": plan.gamma_critical, "certified": None,
            "status": "missing_required", "components": {}, "component_status": {},
            "missing_components": ["target_interface"], "weights": dict(plan.beta),
            "transform": transform.to_dict(),
        }
    d_mu = representation_distance(aligned, derived)
    d_r = jaccard_distance(aligned.active_rules, derived.active_rules)
    d_alpha = activation_distance(aligned, derived)
    d_u = abs(float(aligned.uncertainty) - float(derived.uncertainty))
    d_tau = trace_distance(aligned.trace.as_dict(), derived.trace.as_dict())
    d_l = jaccard_distance(aligned.terms, derived.terms)
    gamma = semantic_disagreement(aligned, derived, plan.beta)
    return {
        "gamma": gamma,
        "gamma_max": plan.gamma_critical,
        "certified": gamma <= plan.gamma_critical,
        "status": "measured",
        "components": {"d_mu": d_mu, "d_R": d_r, "d_alpha": d_alpha, "d_u": d_u, "d_tau": d_tau, "d_L": d_l},
        "component_status": {key: "measured" for key in ("d_mu", "d_R", "d_alpha", "d_u", "d_tau", "d_L")},
        "missing_components": [],
        "weights": dict(plan.beta),
        "transform": transform.to_dict(),
        "aligned": aligned,
    }


def compute_real_pre_interpretability(explanation: ExplanationObject | None, *, plan: ExplainPlan) -> dict[str, Any]:
    """I_pre = exp(-L(E)), computed from a single real explanatory object —
    unlike Γ, this needs only one channel, so it is available whenever any
    local evidence exists at all."""

    if explanation is None:
        return {"i_pre": None, "status": "missing_required", "reason": "no explanatory object could be built (no local evidence with a real score)"}
    return {"i_pre": compute_interpretability_index(explanation, plan.lambda_), "status": "measured", "reason": ""}


# P17: no `compute_real_delta` here — Δ (reduction loss) is not the same
# quantity as linear-reconstruction fidelity (see runtime.py's Γ/Δ/ρ block
# for why conflating them was wrong). `reconstruction_error` stays a
# standalone quality metric in evidence/metrics.py; Δ is populated only
# from a caller-measured, manually supplied `evidence={"reduction": ...}`.
