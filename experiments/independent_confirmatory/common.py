from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "independent_confirmatory"
DATA = ROOT / "data" / "independent_confirmatory"
PROTOCOL = ROOT / "config" / "independent_confirmatory_protocol.json"
AMENDMENT = ROOT / "config" / "independent_confirmatory_protocol_amendment_001.json"
SUMS = ARTIFACTS / "protocol" / "SHA256SUMS"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_protocol() -> dict[str, str]:
    registered = {}
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        digest, path = line.split(maxsplit=1)
        registered[path] = digest
    result = {}
    for path in (PROTOCOL, AMENDMENT):
        relative = str(path.relative_to(ROOT))
        actual = sha256_file(path)
        if registered.get(relative) != actual:
            raise RuntimeError(f"protocol file changed after registration: {relative}")
        result[relative] = actual
    return result


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
