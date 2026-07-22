from __future__ import annotations

from .contracts import RouteArtifact, RouteAssessment, RouteOutcome


CHECKS = (
    ("model_explainer_mismatch", "model_explainer_link"),
    ("stale_calibration", "calibration_registry"),
    ("preprocessing_order_change", "preprocessing_graph"),
    ("feature_schema_incompatibility", "feature_schema"),
    ("cross_model_artifact_mix", "artifact_lineage"),
    ("checksum_corruption", "canonical_store"),
    ("reduction_link_loss", "representation_reducer"),
    ("reference_population_substitution", "reference_population"),
    ("partial_provenance_deletion", "provenance_graph"),
    ("dictionary_or_tokenizer_version_change", "dictionary_registry"),
)


class TypedRouteGuard:
    def __init__(self, *, family_confidence_threshold: float = 0.72) -> None:
        self.family_confidence_threshold = family_confidence_threshold

    def assess(self, artifact: RouteArtifact) -> RouteAssessment:
        violations = self._violations(artifact)
        if not artifact.evidence_complete:
            return RouteAssessment(
                RouteOutcome.INSUFFICIENT,
                False,
                True,
                None,
                0.0,
                ("evidence_channels",),
                tuple(violations),
                ("INSUFFICIENT_EVIDENCE", "AUTOMATIC_CERTIFICATION_DENIED"),
            )
        if not violations:
            return RouteAssessment(RouteOutcome.VALID, False, False, None, 1.0, (), (), ("ROUTE_CERTIFIED",))
        regions = tuple(dict.fromkeys(dict(CHECKS)[item] for item in violations))
        # Multiple overlapping violations lower family confidence and trigger abstention.
        confidence = max(0.35, 0.96 - 0.16 * (len(violations) - 1))
        family = violations[0] if confidence >= self.family_confidence_threshold and len(violations) == 1 else None
        outcome = RouteOutcome.KNOWN_FAULT if family else RouteOutcome.UNKNOWN_FAULT
        irreparable = "checksum_corruption" in violations and "cross_model_artifact_mix" in violations
        return RouteAssessment(
            outcome,
            irreparable,
            not irreparable,
            family,
            confidence,
            regions,
            tuple(violations),
            ((f"KNOWN_FAULT:{family}",) if family else ("UNKNOWN_OR_COMPOSITE_STRUCTURAL_FAULT", "AUTOMATIC_CERTIFICATION_DENIED")),
        )

    @staticmethod
    def _violations(value: RouteArtifact) -> list[str]:
        checks = {
            "model_explainer_mismatch": value.model_id != value.explainer_model_id,
            "stale_calibration": value.model_id != value.calibration_model_id,
            "preprocessing_order_change": value.preprocessing_steps != tuple(sorted(value.preprocessing_steps)),
            "feature_schema_incompatibility": value.feature_schema != value.explainer_feature_schema,
            "cross_model_artifact_mix": len({value.model_id, value.explainer_model_id, value.calibration_model_id}) == 3,
            "checksum_corruption": value.canonical_sha256 != value.observed_sha256,
            "reduction_link_loss": value.reduction_source_id != value.reduction_target_source_id,
            "reference_population_substitution": value.reference_population_id != value.expected_reference_population_id,
            "partial_provenance_deletion": bool(set(value.mandatory_provenance_nodes) - set(value.provenance_nodes)),
            "dictionary_or_tokenizer_version_change": value.dictionary_version != value.expected_dictionary_version,
        }
        return [name for name, _ in CHECKS if checks[name]]
