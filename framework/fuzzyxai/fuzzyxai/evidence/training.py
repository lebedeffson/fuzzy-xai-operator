from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from .contracts import SubgroupAveragingEvidence, TrainingObjectTrace


def find_forgetting_events(epoch_metrics: Sequence[Mapping[str, Any]], *, confidence_drop: float = 0.2) -> list[int]:
    """Return epochs where a learned object becomes wrong or loses confidence."""

    metrics = [dict(item) for item in epoch_metrics]
    events: list[int] = []
    for index in range(1, len(metrics)):
        previous = metrics[index - 1]
        current = metrics[index]
        label_lost = bool(previous.get("correct", False)) and not bool(current.get("correct", False))
        confidence_lost = bool(previous.get("correct", False)) and (
            float(previous.get("confidence", 0.0)) - float(current.get("confidence", 0.0)) >= confidence_drop
        )
        if label_lost or confidence_lost:
            events.append(int(current.get("epoch", index)))
    return events


def build_object_trace(
    object_id: str,
    epoch_metrics: Sequence[Mapping[str, Any]],
    *,
    confidence_drop: float = 0.2,
    provenance: Mapping[str, Any] | None = None,
) -> TrainingObjectTrace:
    """Build a per-object trajectory and locate learned-then-forgotten epochs."""

    metrics = [dict(item) for item in epoch_metrics]
    predicted = [item.get("predicted_class") for item in metrics]
    confidence = [float(item.get("confidence", 0.0)) for item in metrics]
    # An unavailable loss is not a measured zero.
    losses = [None if item.get("loss") is None else float(item["loss"]) for item in metrics]
    embeddings = [list(item.get("embedding", [])) for item in metrics]
    activations = [dict(item.get("rule_activations", {})) for item in metrics]
    correct = [bool(item.get("correct", False)) for item in metrics]
    epochs = [int(item.get("epoch", index)) for index, item in enumerate(metrics)]

    forgetting = find_forgetting_events(metrics, confidence_drop=confidence_drop)
    forgetting_details: list[dict[str, Any]] = []
    by_epoch = {int(item.get("epoch", index)): item for index, item in enumerate(metrics)}
    for event in forgetting:
        position = epochs.index(event)
        previous, current = metrics[position - 1], by_epoch[event]
        correctness_transition = bool(previous.get("correct", False)) and not bool(current.get("correct", False))
        forgetting_details.append({
            "epoch": event,
            "reason": "correctness_transition" if correctness_transition else "confidence_drop",
            "previous_confidence": previous.get("confidence"), "current_confidence": current.get("confidence"),
            "threshold": confidence_drop, "previous_correct": previous.get("correct"), "current_correct": current.get("correct"),
        })
    correct_epochs = [epoch for epoch, flag in zip(epochs, correct) if flag]
    transitions = sum(left != right for left, right in pairwise(correct))
    stability = 1.0 if len(correct) <= 1 else max(0.0, 1.0 - transitions / (len(correct) - 1))
    warnings = [] if metrics else ["training history was not supplied"]
    return TrainingObjectTrace(
        object_id=str(object_id),
        epoch_metrics=metrics,
        predicted_class_by_epoch=predicted,
        confidence_by_epoch=confidence,
        loss_by_epoch=losses,
        embedding_by_epoch=embeddings,
        rule_activation_by_epoch=activations,
        forgetting_events=forgetting,
        stability_score=round(stability, 6),
        first_learned_epoch=correct_epochs[0] if correct_epochs else None,
        last_correct_epoch=correct_epochs[-1] if correct_epochs else None,
        warnings=warnings,
        forgetting_details=forgetting_details,
        loss_status="measured" if metrics and all(item.get("loss") is not None for item in metrics) else "not_measured",
        training_run_id=(provenance or {}).get("training_run_id"),
        model_fingerprint=(provenance or {}).get("model_fingerprint"),
        training_method=(provenance or {}).get("training_method"),
        epoch_source=(provenance or {}).get("epoch_source"),
        final_checkpoint_ref=(provenance or {}).get("final_checkpoint_ref"),
    )


def detect_subgroup_averaging(
    *,
    global_metric: Sequence[float],
    subgroup_metrics: Mapping[str, Sequence[float]],
    subgroup_objects: Mapping[str, Sequence[str]] | None = None,
    subgroup_rule_history: Mapping[str, Sequence[Sequence[str]]] | None = None,
    embedding_spread: Mapping[str, Sequence[float]] | None = None,
    min_global_gain: float = 0.0,
    min_subgroup_drop: float = 0.05,
) -> list[SubgroupAveragingEvidence]:
    """Detect global improvement accompanied by degradation of a subgroup."""

    if len(global_metric) < 2:
        raise ValueError("global_metric requires at least two observations")
    global_change = float(global_metric[-1] - global_metric[0])
    results: list[SubgroupAveragingEvidence] = []
    for subgroup_id, values in subgroup_metrics.items():
        if len(values) < 2:
            continue
        subgroup_change = float(values[-1] - values[0])
        rule_history = list((subgroup_rule_history or {}).get(subgroup_id, []))
        disappeared = sorted(set(rule_history[0]) - set(rule_history[-1])) if len(rule_history) >= 2 else []
        spread = list((embedding_spread or {}).get(subgroup_id, []))
        collapse = float(spread[0] - spread[-1]) if len(spread) >= 2 else None
        averaged = global_change > min_global_gain and (
            subgroup_change <= -min_subgroup_drop or bool(disappeared) or (collapse is not None and collapse > min_subgroup_drop)
        )
        limitations = []
        if not rule_history:
            limitations.append("rule activation history unavailable")
        if not spread:
            limitations.append("embedding spread unavailable")
        objects = list((subgroup_objects or {}).get(subgroup_id, []))
        results.append(
            SubgroupAveragingEvidence(
                subgroup_id=str(subgroup_id),
                size=len(objects),
                global_metric_change=round(global_change, 6),
                subgroup_metric_change=round(subgroup_change, 6),
                minority_recall_change=round(subgroup_change, 6),
                embedding_collapse=None if collapse is None else round(collapse, 6),
                prototype_distance_change=None,
                disappeared_rules=disappeared,
                affected_objects=objects,
                averaged=averaged,
                limitations=limitations,
            )
        )
    return results


@dataclass
class TrainingRunAnalysis:
    """Public analysis object returned by ``FuzzyXAI.observe_training``."""

    traces: Mapping[str, TrainingObjectTrace]
    subgroups: Sequence[SubgroupAveragingEvidence]
    rules: Sequence[Any]
    limitations: Sequence[str] = field(default_factory=tuple)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def find_forgotten_objects(self) -> list[TrainingObjectTrace]:
        return [trace for trace in self.traces.values() if trace.forgetting_events]

    def find_averaged_subgroups(self) -> list[SubgroupAveragingEvidence]:
        return [item for item in self.subgroups if item.averaged]

    def extract_model_rules(self) -> list[Any]:
        return list(self.rules)

    def plot_object_trajectory(self, object_id: str, output_path: str | None = None):
        from fuzzyxai.visualization.training import render_training_trajectory

        if object_id not in self.traces:
            raise KeyError(f"unknown object_id: {object_id}")
        return render_training_trajectory(self.traces[object_id], output_path)
