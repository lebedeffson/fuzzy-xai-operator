from __future__ import annotations

import os

from ..hashing import file_sha256, write_json
from ..paths import ARTIFACT_ROOT
from .opening_counter import initialize


def seal_current_inputs() -> dict:
    sealed = ARTIFACT_ROOT / "data" / "sealed"
    private = ARTIFACT_ROOT / "private" / "sealed"
    if not (sealed / "cases.jsonl").exists() or not (private / "gold.jsonl").exists():
        raise FileNotFoundError("sealed inputs are not generated")
    manifest = {
        "public_cases_sha256": file_sha256(sealed / "cases.jsonl"),
        "private_gold_sha256": file_sha256(private / "gold.jsonl"),
        "private_transactions_sha256": file_sha256(private / "transactions.jsonl"),
        "vault_release_policy": "private files excluded from release archives",
        "mode": "0600",
    }
    for path in private.iterdir():
        os.chmod(path, 0o600)
    write_json(ARTIFACT_ROOT / "sealed" / "sealed_manifest.json", manifest)
    initialize(ARTIFACT_ROOT / "sealed" / "opening_record.json")
    return manifest

