from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fuzzyxai.experiments.h10_c7_model_lock import (
    snapshot_path,
    snapshot_record,
    verify_model_weight_lock,
)


def _fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    cache = tmp_path / "hub"
    name = "fixture/code-encoder"
    revision = "a" * 40
    snapshot = snapshot_path(cache, name, revision)
    snapshot.mkdir(parents=True)
    for filename, payload in {
        "config.json": b'{"model_type":"roberta"}\n',
        "tokenizer_config.json": b"{}\n",
        "vocab.json": b'{"token":0}\n',
        "merges.txt": b"#version: 0.2\n",
        "pytorch_model.bin": b"registered weights",
    }.items():
        (snapshot / filename).write_bytes(payload)
    item = {
        "backend": "local_transformer",
        "model_name": name,
        "repository_revision": revision,
        "revision": revision,
        "weights_sha256": "",
    }
    record = snapshot_record(snapshot, item)
    item["weights_sha256"] = record["weights_sha256"]
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"dense_encoders": [item], "cross_encoders": []}),
        encoding="utf-8",
    )
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "method_commit": "b" * 40,
                "network_allowed_during_scoring": False,
                "models": [record],
            }
        ),
        encoding="utf-8",
    )
    return registry, lock, snapshot


def test_model_weight_lock_verifies_hashes_and_read_only_snapshot(
    tmp_path: Path,
) -> None:
    registry, lock, snapshot = _fixture(tmp_path)
    for path in snapshot.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o444)
    os.chmod(snapshot, 0o555)
    try:
        result = verify_model_weight_lock(registry, lock)
    finally:
        os.chmod(snapshot, 0o755)
        for path in snapshot.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o644)
    assert result["status"] == "H10_C7_MODEL_WEIGHT_LOCK_PASS"
    assert result["model_count"] == 1


def test_model_weight_lock_rejects_modified_weights(tmp_path: Path) -> None:
    registry, lock, snapshot = _fixture(tmp_path)
    (snapshot / "pytorch_model.bin").write_bytes(b"modified")
    with pytest.raises(ValueError, match="snapshot mismatch"):
        verify_model_weight_lock(
            registry,
            lock,
            require_read_only=False,
        )
