"""Controlled and realistic replay fault taxonomy."""

from __future__ import annotations

from dataclasses import dataclass


FAULT_FAMILIES = (
    "model_version", "preprocessing_version", "feature_mapping", "category_dictionary", "tokenization",
    "image_normalization", "sampling_rate", "calibration_missing", "calibration_stale", "reference_population",
    "provenance_edge", "explainer_model_pair", "artifact_corruption", "artifact_hash", "model_card_stale",
    "schema_unsupported", "reduction_loss", "required_channel", "explain_plan", "nondeterminism",
    "reference_shift", "cache_corruption", "feature_order", "feature_unit", "class_dictionary",
    "background_set", "segmentation_version", "mask_resolution", "embedding_version", "window_alignment",
    "timezone", "missingness_contract", "grouping_key", "label_mapping", "probability_shape",
    "output_semantics", "rule_conflict", "deployment_envelope", "data_quality", "audit_signature",
)


@dataclass(frozen=True)
class FaultTemplate:
    template_id: str
    family: str
    source: str
    diagnosable: bool = True


def fault_library() -> tuple[FaultTemplate, ...]:
    return tuple(FaultTemplate(f"FT-{index:02d}", family, f"route:{family}") for index, family in enumerate(FAULT_FAMILIES, 1))


def compositional_faults() -> tuple[tuple[str, ...], ...]:
    templates = fault_library()
    pairs = tuple((templates[index].template_id, templates[index + 1].template_id) for index in range(0, 20, 2))
    triples = tuple(tuple(item.template_id for item in templates[index:index + 3]) for index in range(20, 30, 3))
    return pairs + triples
