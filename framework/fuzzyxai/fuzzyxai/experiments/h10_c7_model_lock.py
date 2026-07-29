from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any

TOKENIZER_FILES = {
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_sha256(entries: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for relative_path, file_digest in sorted(entries):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_digest))
    return digest.hexdigest()


def snapshot_path(cache_root: Path, model_name: str, revision: str) -> Path:
    repository = "models--" + model_name.replace("/", "--")
    return cache_root / repository / "snapshots" / revision


def snapshot_record(
    snapshot: Path,
    item: dict[str, Any],
) -> dict[str, Any]:
    if not snapshot.is_dir():
        raise ValueError(f"missing pinned snapshot: {snapshot}")
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"empty pinned snapshot: {snapshot}")
    hashes = {
        path.relative_to(snapshot).as_posix(): file_sha256(path)
        for path in files
    }
    if "config.json" not in hashes:
        raise ValueError(f"missing config.json: {snapshot}")
    tokenizer_entries = [
        (name, digest)
        for name, digest in hashes.items()
        if Path(name).name in TOKENIZER_FILES
    ]
    if not tokenizer_entries:
        raise ValueError(f"missing tokenizer files: {snapshot}")
    weight_names = [name for name in hashes if Path(name).name in WEIGHT_FILES]
    if len(weight_names) != 1:
        raise ValueError(
            f"expected one runtime weight file in {snapshot}, found {weight_names}"
        )
    weight_name = weight_names[0]
    return {
        "name": str(item["model_name"]),
        "revision": str(item["revision"]),
        "repository_revision": str(item["repository_revision"]),
        "local_path": str(snapshot.resolve()),
        "snapshot_hash_algorithm": "sha256_tree_v1",
        "snapshot_sha256": aggregate_sha256(list(hashes.items())),
        "tokenizer_sha256": aggregate_sha256(tokenizer_entries),
        "config_sha256": hashes["config.json"],
        "weights_file": weight_name,
        "weights_sha256": hashes[weight_name],
        "file_count": len(hashes),
        "files": hashes,
    }


def _is_read_only(path: Path) -> bool:
    return not stat.S_IMODE(path.stat().st_mode) & 0o222


def verify_model_weight_lock(
    registry_path: Path,
    lock_path: Path,
    *,
    require_read_only: bool = True,
) -> dict[str, object]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    registry_items = {
        str(item["model_name"]): item
        for group in ("dense_encoders", "cross_encoders")
        for item in registry.get(group, [])
    }
    lock_items = {
        str(item["name"]): item for item in lock.get("models", [])
    }
    if registry_items.keys() != lock_items.keys():
        raise ValueError("model registry and model lock contain different models")
    verified = []
    for name, item in registry_items.items():
        expected = lock_items[name]
        for field in ("revision", "repository_revision", "weights_sha256"):
            if str(item[field]) != str(expected[field]):
                raise ValueError(f"{name}: model lock mismatch for {field}")
        snapshot = Path(str(expected["local_path"]))
        observed = snapshot_record(snapshot, item)
        for field in (
            "snapshot_sha256",
            "tokenizer_sha256",
            "config_sha256",
            "weights_sha256",
            "file_count",
            "files",
        ):
            if observed[field] != expected[field]:
                raise ValueError(f"{name}: model snapshot mismatch for {field}")
        if require_read_only:
            paths = [snapshot, *snapshot.rglob("*")]
            writable = [
                str(path)
                for path in paths
                if path.exists() and not _is_read_only(path.resolve())
            ]
            if writable:
                raise ValueError(
                    f"{name}: model snapshot is writable: {writable[:3]}"
                )
        verified.append(name)
    return {
        "status": "H10_C7_MODEL_WEIGHT_LOCK_PASS",
        "method_commit": lock["method_commit"],
        "network_allowed_during_scoring": False,
        "model_count": len(verified),
        "verified_models": sorted(verified),
    }
