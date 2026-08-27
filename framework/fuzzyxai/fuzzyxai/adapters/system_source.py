"""Domain-neutral providers for the system-operator source interface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fuzzyxai.adapters.model import ModelPrediction
from fuzzyxai.core.explanation_object import Rule, Trace
from fuzzyxai.system_semantics import SystemSourceEvidence


def source_term_id(interface_id: str, class_value: Any) -> str:
    return f"{interface_id}:class:{class_value}"


def source_rule_id(class_value: Any) -> str:
    return f"source_class_{class_value}"


def _row(values: Any) -> list[Any]:
    current = values
    while isinstance(current, list) and current and isinstance(current[0], list):
        current = current[0]
    return list(current) if isinstance(current, list) else []


def derive_system_source_evidence(
    *,
    object_id: str,
    model_fingerprint: str,
    prediction: ModelPrediction,
    internal_evidence: Mapping[str, Any],
    source_interface_id: str,
    risk_class: Any,
    trace: Trace,
    model_trace: Trace | None,
    source_refs: tuple[str, ...] = (),
) -> SystemSourceEvidence:
    """Adapt native votes or class probabilities to one generic source contract.

    This provider knows model channels. The system operator itself only consumes
    :class:`SystemSourceEvidence` and is therefore independent of RF and domain
    class names.
    """

    classes = _row(dict(prediction.metadata).get("classes"))
    probabilities = [float(value) for value in _row(prediction.probabilities)]
    raw_votes = internal_evidence.get("ensemble_votes")
    votes: list[Any] = []
    if isinstance(raw_votes, list):
        for item in raw_votes:
            votes.append(item[0] if isinstance(item, (list, tuple)) and item else item)
    if not classes:
        classes = sorted(set(votes), key=str) if votes else list(range(len(probabilities)))
    if risk_class not in classes:
        raise ValueError(f"registered system risk class {risk_class!r} is absent from model classes")

    normalized_probabilities: list[float] = []
    if probabilities:
        if len(probabilities) != len(classes):
            raise ValueError("model class probabilities do not match registered class labels")
        total = sum(probabilities)
        if total <= 0:
            raise ValueError("model class probabilities must have positive sum")
        normalized_probabilities = [item / total for item in probabilities]

    if votes:
        vote_indicators = [1.0 if item == risk_class else 0.0 for item in votes]
        vote_proportion = sum(vote_indicators) / len(vote_indicators)
        vote_disagreement = float((sum((item - vote_proportion) ** 2 for item in vote_indicators) / len(vote_indicators)) ** 0.5)
        counts = {str(label): sum(1 for item in votes if item == label) for label in classes}
        provider = "native_ensemble_votes"
        if normalized_probabilities:
            activations_by_class = dict(zip(classes, normalized_probabilities))
            value = float(activations_by_class[risk_class])
            representation_semantics = "class_probability"
            representation_source = "prediction.probabilities"
        else:
            activations_by_class = {label: counts[str(label)] / len(votes) for label in classes}
            value = float(vote_proportion)
            representation_semantics = "vote_proportion"
            representation_source = "ensemble_votes_fallback"
        uncertainty_inputs: dict[str, Any] = {
            "ensemble_vote_standard_deviation": vote_disagreement,
            "vote_indicator_risk_class": risk_class,
            "vote_indicators": vote_indicators,
            "votes": list(votes),
            "vote_count": len(votes),
            "vote_counts": counts,
            "vote_proportions": {label: count / len(votes) for label, count in counts.items()},
        }
        if normalized_probabilities:
            uncertainty_inputs.update({"probabilities": normalized_probabilities, "classes": list(classes)})
    elif normalized_probabilities:
        activations_by_class = dict(zip(classes, normalized_probabilities))
        value = float(activations_by_class[risk_class])
        provider = "native_class_probabilities"
        representation_semantics = "class_probability"
        representation_source = "prediction.probabilities"
        uncertainty_inputs = {"probabilities": normalized_probabilities, "classes": list(classes)}
    else:
        raise ValueError("system source provider requires native ensemble votes or class probabilities")

    terms = tuple(source_term_id(source_interface_id, label) for label in classes)
    rules = tuple(
        Rule(source_rule_id(label), {"model_class": str(label)}, str(label))
        for label in classes
    )
    activations = {
        source_rule_id(label): float(activations_by_class.get(label, 0.0))
        for label in classes
    }
    source_trace = model_trace or Trace(
        id=str(object_id),
        version=model_fingerprint[:12],
        timestamp=trace.timestamp,
        params={"provider": provider, "risk_class": risk_class},
        source=provider,
        checksum=f"{object_id}:{provider}:{value:.12g}",
    )
    return SystemSourceEvidence(
        source_interface_id=source_interface_id,
        terms=terms,
        representation_value=value,
        representation_label=f"{representation_semantics}:{risk_class}",
        rules=rules,
        activations=activations,
        model_uncertainty_inputs=uncertainty_inputs,
        trace=source_trace,
        source_refs=tuple(dict.fromkeys((*source_refs, provider))),
        metadata={
            "provider": provider,
            "classes": list(classes),
            "risk_class": risk_class,
            "risk_coordinate": value,
            "representation_semantics": representation_semantics,
            "representation_source": representation_source,
        },
    )
