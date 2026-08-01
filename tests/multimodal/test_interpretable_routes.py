from __future__ import annotations

from fuzzyxai.diagnostics import DiagnosticValidator
from fuzzyxai.multimodal import MODALITIES, build_route_case


def test_all_modalities_build_valid_routes() -> None:
    validator = DiagnosticValidator()
    for modality in MODALITIES:
        case = build_route_case(modality, 1)
        assert validator.validate(case.graph).valid
        assert case.graph.metadata["interpretability_scope"] == "extracted_feature_space_only"


def test_controlled_violations_are_detected_and_localized() -> None:
    validator = DiagnosticValidator()
    for modality in MODALITIES:
        for violation in ("extractor_checksum", "model_version", "sample_identity", "unknown_relation"):
            case = build_route_case(modality, 2, violation=violation)
            result = validator.validate(case.graph)
            assert not result.valid
            assert case.expected_source in {
                source for issue in result.issues for source in issue.source_nodes
            }


def test_valid_hash_is_restored_by_registered_reconstruction() -> None:
    for modality in MODALITIES:
        broken = build_route_case(modality, 7, violation="model_version")
        valid = build_route_case(modality, 7)
        assert broken.canonical_valid_hash == valid.graph.trace_sha256
