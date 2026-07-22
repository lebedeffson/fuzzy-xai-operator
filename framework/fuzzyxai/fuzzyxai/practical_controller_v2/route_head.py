"""Route-certification risk head."""

from __future__ import annotations

from fuzzyxai.audit_certificate import AuditFeatureVector

from .calibration import CalibratedRiskHead

ROUTE_FEATURES = ("certificate_exists", "certificate_coverage", "unsatisfied_contracts", "weighted_contract_severity", "minimal_cut_size", "path_redundancy", "provenance_completeness", "canonical_integrity")


def route_feature_map(features: AuditFeatureVector) -> dict[str, float]:
    return {
        "certificate_exists": 1.0 - features.certificate_exists,
        "certificate_coverage": 1.0 - features.certificate_coverage,
        "unsatisfied_contracts": min(1.0, features.number_of_unsatisfied_contracts / 10.0),
        "weighted_contract_severity": min(1.0, features.weighted_contract_severity / 10.0),
        "minimal_cut_size": min(1.0, features.minimal_cut_size / 10.0),
        "path_redundancy": features.path_redundancy,
        "provenance_completeness": 1.0 - features.provenance_completeness,
        "canonical_integrity": 1.0 - features.canonical_integrity,
    }


def estimate_route_risk(head: CalibratedRiskHead, features: AuditFeatureVector) -> float:
    if head.target_name != "route_not_certifiable":
        raise ValueError("route head target must be route_not_certifiable")
    return head.predict(route_feature_map(features))
