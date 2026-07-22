"""Structural open-set features from certificates and diagnostic graphs."""

from __future__ import annotations

from fuzzyxai.audit_certificate import AuditFeatureVector

from .contracts import StructuralObservation


FEATURE_REGIONS = {
    "known_template_distance": "fault_library",
    "type_contract_violation": "schema_and_types",
    "unexpected_path_score": "route_graph",
    "certificate_depth_anomaly": "certificate",
    "unknown_composition_score": "contract_composition",
    "known_class_uncertainty": "fault_classifier",
    "structural_energy": "route_graph",
    "graph_embedding_distance": "route_graph",
}


def structural_observation(
    observation_id: str,
    audit: AuditFeatureVector,
    *,
    known_template_distance: float,
    type_contract_violation: float,
    unexpected_path_score: float,
    unknown_composition_score: float,
    known_class_uncertainty: float,
    graph_embedding_distance: float,
    missing_channels: tuple[str, ...] = (),
    partition: str = "development",
    source_is_oof: bool = True,
) -> StructuralObservation:
    features = {
        "known_template_distance": known_template_distance,
        "type_contract_violation": type_contract_violation,
        "unexpected_path_score": unexpected_path_score,
        "certificate_depth_anomaly": min(1.0, audit.certificate_depth / 10.0),
        "unknown_composition_score": unknown_composition_score,
        "known_class_uncertainty": known_class_uncertainty,
        "structural_energy": min(1.0, (audit.weighted_contract_severity + audit.number_of_unsatisfied_contracts) / 10.0),
        "graph_embedding_distance": graph_embedding_distance,
    }
    return StructuralObservation(observation_id, features, FEATURE_REGIONS, missing_channels, source_is_oof, partition)
