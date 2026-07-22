from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaultSpec:
    parent: str
    leaf: str
    fields: tuple[str, ...]
    source_nodes: tuple[str, ...]
    repair_action: str


# This is the evaluated auditor taxonomy. It is intentionally separate from
# experiments.h10.oracle_v19, which supplies adjudication truth without imports
# from this package.
FAULT_SPECS = (
    FaultSpec("artifact_integrity", "hash_corruption", ("artifact_sha256",), ("canonical_artifact",), "restore_verified_artifact"),
    FaultSpec("artifact_integrity", "missing_artifact", ("artifact_uri",), ("artifact_store",), "reload_missing_artifact"),
    FaultSpec("artifact_integrity", "version_mismatch", ("model_version", "artifact_model_id", "model_id"), ("model_registry",), "align_component_versions"),
    FaultSpec("semantic_compatibility", "model_explainer_mismatch", ("model_family", "explainer_model_family", "explainer_version"), ("compatibility_registry",), "load_compatible_explainer"),
    FaultSpec("semantic_compatibility", "preprocessing_mismatch", ("preprocessing_signature", "dependency_digest", "canonical_source_id"), ("preprocessor_registry",), "restore_registered_preprocessing"),
    FaultSpec("semantic_compatibility", "dictionary_mismatch", ("dictionary_version", "canonical_source_id", "dependency_digest"), ("dictionary_registry",), "restore_registered_dictionary"),
    FaultSpec("reference_context", "wrong_reference_population", ("reference_population", "deployment_context", "calibration_version"), ("reference_registry",), "restore_reference_population"),
    FaultSpec("reference_context", "stale_calibration", ("calibration_version", "calibration_age_days", "reference_population"), ("calibration_registry",), "refresh_calibration"),
    FaultSpec("reference_context", "deployment_context_mismatch", ("deployment_context", "reference_population", "calibration_version"), ("deployment_registry",), "restore_deployment_context"),
    FaultSpec("provenance", "missing_source", ("source_uri", "dependency_digest", "canonical_source_id"), ("provenance_registry",), "restore_source_reference"),
    FaultSpec("provenance", "broken_dependency", ("dependency_digest", "canonical_source_id", "artifact_sha256"), ("dependency_registry",), "rebuild_dependency_link"),
    FaultSpec("provenance", "cross_model_artifact_mix", ("artifact_model_id", "model_id", "model_version"), ("artifact_binding_registry",), "replace_cross_model_artifact"),
    FaultSpec("reduction", "excessive_information_loss", ("reduction_loss", "projection_type", "canonical_source_id"), ("reduction_registry",), "select_loss_bounded_projection"),
    FaultSpec("reduction", "unsupported_projection", ("projection_type", "canonical_source_id", "reduction_loss"), ("reduction_registry",), "select_supported_projection"),
    FaultSpec("reduction", "lost_canonical_link", ("canonical_source_id", "artifact_uri", "artifact_sha256"), ("canonical_registry",), "restore_canonical_link"),
)

FAULT_TAXONOMY: dict[str, tuple[str, ...]] = {}
for _spec in FAULT_SPECS:
    FAULT_TAXONOMY.setdefault(_spec.parent, ())
    FAULT_TAXONOMY[_spec.parent] += (_spec.leaf,)

SPEC_BY_LEAF = {spec.leaf: spec for spec in FAULT_SPECS}
FIELD_TO_SPECS = {field: tuple(spec for spec in FAULT_SPECS if field in spec.fields) for spec in FAULT_SPECS for field in spec.fields}
