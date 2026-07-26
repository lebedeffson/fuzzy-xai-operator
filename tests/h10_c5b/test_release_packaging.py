from __future__ import annotations

from pathlib import Path

from scripts.build_framework_release import include_in_source_release


def test_h10_c5b_and_h9_v2_evidence_is_in_source_release() -> None:
    paths = (
        "protocol/h10_c5b_repository_grounded/H10_C5B_PROTOCOL_LOCK.json",
        "protocol/h10_c5b_repository_grounded/H10_C5B_RUNTIME_AVAILABILITY_AMENDMENT.json",
        "protocol/h10_c5b_repository_grounded/H10_C5B_RUNTIME_ENVIRONMENT_AMENDMENT.json",
        "protocol/h10_c5b_repository_grounded/H10_C5B_RUNTIME_STORAGE_AMENDMENT.json",
        "protocol/h10_c5b_repository_grounded/H10_C5B_RUNTIME_PYTEST_ENVIRONMENT_AMENDMENT.json",
        "protocol/h10_c5b_repository_grounded/METHOD_LOCK.json",
        "protocol/h10_c5b_repository_grounded/CONTAINER_IMAGE_REGISTRY_SCHEMA.json",
        "protocol/h10_c5b_repository_grounded/RUNTIME_COLLECTION_RUNBOOK.md",
        ".github/workflows/h10-c5b-runtime-collection.yml",
        "protocol/h9_e2e_v2/H9_E2E_V2_PROTOCOL_LOCK.json",
        "reports/h10_c5b/REPOSITORY_GROUNDED_TRANSFER.md",
        "reports/h10_c5b/RUNTIME_COLLECTION_READINESS.json",
        "reports/h10_c5b/RUNTIME_COLLECTION_SHA256SUMS",
        "reports/h9_e2e_v2/OPTIMIZED_EVIDENCE_PATH.md",
        "results/h10_c5b/SHA256SUMS",
        "results/h9_e2e_v2/SHA256SUMS",
    )
    assert all(
        include_in_source_release(Path(path), set())
        for path in paths
    )
