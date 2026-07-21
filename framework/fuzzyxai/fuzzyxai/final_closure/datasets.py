"""Dataset registry and leakage audit for sealed confirmation."""

from __future__ import annotations

import re
from dataclasses import dataclass


SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SealedDataset:
    dataset_id: str
    modality: str
    source: str
    license: str
    download_sha256: str
    preprocessing_sha256: str
    train_ids_sha256: str
    development_ids_sha256: str
    test_ids_sha256: str
    grouping_key: str | None
    label_vault_sha256: str
    used_in_formative_tuning: bool = False

    def __post_init__(self) -> None:
        if self.modality not in {"tabular", "image", "text", "timeseries"}:
            raise ValueError("unsupported confirmatory modality")
        digests = (
            self.download_sha256,
            self.preprocessing_sha256,
            self.train_ids_sha256,
            self.development_ids_sha256,
            self.test_ids_sha256,
            self.label_vault_sha256,
        )
        if any(not SHA256.fullmatch(value) for value in digests):
            raise ValueError("all sealed dataset identities must be SHA256 digests")
        if len({self.train_ids_sha256, self.development_ids_sha256, self.test_ids_sha256}) != 3:
            raise ValueError("train, development and test identity hashes must differ")
        if self.used_in_formative_tuning:
            raise ValueError("confirmatory dataset was used in formative tuning")


def audit_registry(
    datasets: tuple[SealedDataset, ...],
    *,
    formative_dataset_ids: set[str],
    formative_hashes: set[str],
    oof_object_hashes: set[str],
    sealed_test_object_hashes: set[str],
    tuning_runner_can_read_test_labels: bool,
) -> dict[str, object]:
    blockers: list[str] = []
    counts: dict[str, int] = {}
    dataset_ids: set[str] = set()
    download_hashes: set[str] = set()
    for dataset in datasets:
        if dataset.dataset_id in dataset_ids:
            blockers.append(f"DUPLICATE_DATASET_ID:{dataset.dataset_id}")
        dataset_ids.add(dataset.dataset_id)
        if dataset.download_sha256 in download_hashes:
            blockers.append(f"DUPLICATE_DATASET_CONTENT:{dataset.dataset_id}")
        download_hashes.add(dataset.download_sha256)
        counts[dataset.modality] = counts.get(dataset.modality, 0) + 1
        if dataset.dataset_id in formative_dataset_ids:
            blockers.append(f"FORMATIVE_DATASET_ID_REUSE:{dataset.dataset_id}")
        if dataset.download_sha256 in formative_hashes or dataset.preprocessing_sha256 in formative_hashes:
            blockers.append(f"FORMATIVE_DATASET_HASH_REUSE:{dataset.dataset_id}")
    required = {"tabular": 2, "image": 1, "text": 1, "timeseries": 1}
    for modality, minimum in required.items():
        if counts.get(modality, 0) < minimum:
            blockers.append(f"DATASET_COUNT:{modality}:{counts.get(modality, 0)}/{minimum}")
    invalid_oof_hashes = sum(not SHA256.fullmatch(value) for value in oof_object_hashes)
    invalid_test_hashes = sum(not SHA256.fullmatch(value) for value in sealed_test_object_hashes)
    if not oof_object_hashes:
        blockers.append("OOF_IDENTITIES_MISSING")
    if not sealed_test_object_hashes:
        blockers.append("SEALED_TEST_IDENTITIES_MISSING")
    if invalid_oof_hashes:
        blockers.append(f"INVALID_OOF_ID_HASH:{invalid_oof_hashes}")
    if invalid_test_hashes:
        blockers.append(f"INVALID_TEST_ID_HASH:{invalid_test_hashes}")
    overlap = oof_object_hashes & sealed_test_object_hashes
    if overlap:
        blockers.append(f"OOF_TEST_ID_OVERLAP:{len(overlap)}")
    if tuning_runner_can_read_test_labels:
        blockers.append("TEST_LABEL_ACCESSIBLE_TO_TUNING_RUNNER")
    return {"status": "pass" if not blockers else "blocked", "blockers": blockers, "modality_counts": counts}
