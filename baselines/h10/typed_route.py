from __future__ import annotations

from .common import BaselineResult, changed, missing


PARENT_BY_FIELD = {
    "artifact_sha256": "artifact_integrity",
    "artifact_uri": "artifact_integrity",
    "model_version": "artifact_integrity",
    "explainer_version": "artifact_integrity",
    "model_family": "semantic_compatibility",
    "explainer_model_family": "semantic_compatibility",
    "preprocessing_signature": "semantic_compatibility",
    "dictionary_version": "semantic_compatibility",
    "reference_population": "reference_context",
    "calibration_version": "reference_context",
    "calibration_age_days": "reference_context",
    "deployment_context": "reference_context",
    "source_uri": "provenance",
    "dependency_digest": "provenance",
    "artifact_model_id": "provenance",
    "model_id": "provenance",
    "reduction_loss": "reduction",
    "projection_type": "reduction",
    "canonical_source_id": "reduction",
}


class TypedRouteBaseline:
    name = "typed_route"

    def diagnose(self, route: object) -> BaselineResult:
        absent = missing(route)
        active = tuple(dict.fromkeys(absent + changed(route)))
        if not active:
            return BaselineResult("valid", confidence=1.0)
        parents = {PARENT_BY_FIELD[field] for field in active if field in PARENT_BY_FIELD}
        parent = next(iter(parents)) if len(parents) == 1 else None
        return BaselineResult(
            "insufficient_evidence" if absent else "invalid",
            parent_family=parent,
            source_nodes=active,
            unknown=not bool(parents),
            abstained=True,
            confidence=0.7 if parent else 0.0,
        )
