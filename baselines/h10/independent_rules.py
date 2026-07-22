from __future__ import annotations

from .common import BaselineResult, changed, missing


# Independently authored field rules. This module does not import the H10
# taxonomy, classifier, localizer, cut solver, or repair planner.
RULES = {
    "artifact_sha256": ("artifact_integrity", "hash_corruption", "canonical_artifact"),
    "artifact_uri": ("artifact_integrity", "missing_artifact", "artifact_store"),
    "model_version": ("artifact_integrity", "version_mismatch", "model_registry"),
    "explainer_version": ("semantic_compatibility", "model_explainer_mismatch", "compatibility_registry"),
    "model_family": ("semantic_compatibility", "model_explainer_mismatch", "compatibility_registry"),
    "explainer_model_family": ("semantic_compatibility", "model_explainer_mismatch", "compatibility_registry"),
    "preprocessing_signature": ("semantic_compatibility", "preprocessing_mismatch", "preprocessor_registry"),
    "dictionary_version": ("semantic_compatibility", "dictionary_mismatch", "dictionary_registry"),
    "reference_population": ("reference_context", "wrong_reference_population", "reference_registry"),
    "calibration_version": ("reference_context", "stale_calibration", "calibration_registry"),
    "calibration_age_days": ("reference_context", "stale_calibration", "calibration_registry"),
    "deployment_context": ("reference_context", "deployment_context_mismatch", "deployment_registry"),
    "source_uri": ("provenance", "missing_source", "provenance_registry"),
    "dependency_digest": ("provenance", "broken_dependency", "dependency_registry"),
    "artifact_model_id": ("provenance", "cross_model_artifact_mix", "artifact_binding_registry"),
    "model_id": ("provenance", "cross_model_artifact_mix", "artifact_binding_registry"),
    "reduction_loss": ("reduction", "excessive_information_loss", "reduction_registry"),
    "projection_type": ("reduction", "unsupported_projection", "reduction_registry"),
    "canonical_source_id": ("reduction", "lost_canonical_link", "canonical_registry"),
}


class IndependentRulesBaseline:
    name = "independent_if_else"

    def diagnose(self, route: object) -> BaselineResult:
        absent = missing(route)
        differences = tuple(field for field in changed(route) if field in RULES)
        active = tuple(dict.fromkeys(absent + differences))
        if not active:
            return BaselineResult("valid", confidence=1.0)
        matches = [RULES[field] for field in active if field in RULES]
        family = matches[0][0] if matches and all(item[0] == matches[0][0] for item in matches) else None
        leaf = matches[0][1] if matches and all(item[1] == matches[0][1] for item in matches) else None
        sources = tuple(dict.fromkeys(item[2] for item in matches))
        return BaselineResult(
            "insufficient_evidence" if absent else "invalid",
            family,
            leaf,
            sources,
            active,
            sources,
            unknown=not bool(matches),
            abstained=leaf is None,
            confidence=1.0 if leaf else 0.5,
        )
