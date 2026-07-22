from __future__ import annotations

from fuzzyxai.audit_h10.taxonomy import FAULT_SPECS, FAULT_TAXONOMY


def test_taxonomy_has_registered_parent_and_leaf_levels() -> None:
    assert set(FAULT_TAXONOMY) == {
        "artifact_integrity",
        "semantic_compatibility",
        "reference_context",
        "provenance",
        "reduction",
    }
    assert len(FAULT_SPECS) == 15
    assert all(len(leaves) >= 3 for leaves in FAULT_TAXONOMY.values())
