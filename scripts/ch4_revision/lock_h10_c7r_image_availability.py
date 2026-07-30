#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_MAX_COMPRESSED_BYTES = 2 * 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(row: dict[str, object]) -> dict[str, object]:
    tag = str(row["container_image_tag"])
    completed = subprocess.run(
        ["docker", "manifest", "inspect", "--verbose", tag],
        capture_output=True,
        check=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(completed.stdout)
    manifest = payload.get("SchemaV2Manifest", payload)
    layers = manifest.get("layers", [])
    descriptor = payload.get("Descriptor", {})
    compressed_bytes = sum(int(layer.get("size", 0)) for layer in layers)
    return {
        "incident_id": str(row["incident_id"]),
        "container_image_tag": tag,
        "manifest_digest": descriptor.get("digest"),
        "compressed_bytes": compressed_bytes,
        "max_layer_bytes": max(
            (int(layer.get("size", 0)) for layer in layers),
            default=0,
        ),
    }


def lock(
    registry_path: Path,
    output: Path,
    *,
    max_compressed_bytes: int,
    workers: int,
) -> dict[str, object]:
    registry = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        entries = list(executor.map(inspect_image, registry))
    entries.sort(key=lambda row: str(row["incident_id"]))
    for entry in entries:
        available = (
            int(entry["compressed_bytes"]) <= max_compressed_bytes
            and int(entry["max_layer_bytes"]) <= max_compressed_bytes
        )
        entry["availability_status"] = (
            "AVAILABLE_WITHIN_RUNNER_IMAGE_BUDGET"
            if available
            else "RUNTIME_INFRASTRUCTURE_UNAVAILABLE_IMAGE_BUDGET"
        )
    result = {
        "artifact": "H10_C7R_IMAGE_AVAILABILITY_LOCK",
        "created_at": datetime.now(UTC).isoformat(),
        "created_before_gold_opening": True,
        "gold_used": False,
        "scientific_scoring_started": False,
        "registry_path": str(registry_path),
        "registry_sha256": sha256(registry_path),
        "max_compressed_bytes": max_compressed_bytes,
        "max_layer_bytes": max_compressed_bytes,
        "pull_timeout_seconds": 1800,
        "rule": (
            "A digest-pinned image is runtime-available only when both its "
            "total compressed size and largest compressed layer do not exceed "
            "the fixed runner budget."
        ),
        "entries": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-compressed-bytes",
        type=int,
        default=DEFAULT_MAX_COMPRESSED_BYTES,
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = lock(
        args.runtime_registry.resolve(),
        args.output.resolve(),
        max_compressed_bytes=args.max_compressed_bytes,
        workers=args.workers,
    )
    unavailable = sum(
        entry["availability_status"]
        == "RUNTIME_INFRASTRUCTURE_UNAVAILABLE_IMAGE_BUDGET"
        for entry in result["entries"]
    )
    print(
        json.dumps(
            {
                "available": len(result["entries"]) - unavailable,
                "max_compressed_bytes": result["max_compressed_bytes"],
                "status": "H10_C7R_IMAGE_AVAILABILITY_LOCKED",
                "unavailable": unavailable,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
