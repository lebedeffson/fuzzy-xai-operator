from __future__ import annotations

from pathlib import Path

from .hashing import file_sha256, write_json


def build_manifest(paths: list[Path], output: Path) -> dict:
    value = {"files": [{"path": str(path), "sha256": file_sha256(path)} for path in sorted(paths)]}
    write_json(output, value)
    return value

