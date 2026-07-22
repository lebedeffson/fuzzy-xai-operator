from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "operational_audit_v16"
DATA = ROOT / "data" / "operational_audit_v16"
PROTOCOL = ROOT / "config" / "operational_audit_v16_protocol.json"
AMENDMENT = ROOT / "config" / "operational_audit_v16_protocol_amendment_001.json"
SUMS = ARTIFACTS / "protocol" / "SHA256SUMS"
LOCK = ARTIFACTS / "lock" / "protocol_lock.json"
OPENING = ARTIFACTS / "opening" / "opening_record.json"


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


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def verify_protocol() -> None:
    expected = {line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0] for line in SUMS.read_text().splitlines()}
    for path in (PROTOCOL, AMENDMENT):
        relative = str(path.relative_to(ROOT))
        if expected.get(relative) != sha256_file(path):
            raise RuntimeError(f"protocol hash mismatch: {relative}")


def unit(value: str, salt: str) -> float:
    return int.from_bytes(hashlib.sha256(f"{salt}:{value}".encode()).digest()[:8], "big") / 2**64
