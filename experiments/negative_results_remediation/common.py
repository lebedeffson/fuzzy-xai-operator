from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"
ARTIFACTS = ROOT / "artifacts" / "negative_results_remediation"
PROTOCOL = CONFIG / "negative_remediation_protocol.json"
PROTOCOL_SUMS = ARTIFACTS / "protocol" / "SHA256SUMS"
FROZEN_NEGATIVE_CLAIMS = {
    "H3-original": "not_supported",
    "H5-P-original": "not_supported",
    "H6-general": "not_supported",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n")
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_protocol() -> str:
    expected = None
    for line in PROTOCOL_SUMS.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        if relative == "config/negative_remediation_protocol.json":
            expected = digest
            break
    if expected is None:
        raise RuntimeError("protocol hash is absent from SHA256SUMS")
    actual = sha256_file(PROTOCOL)
    if actual != expected:
        raise RuntimeError(f"protocol changed after freeze: expected {expected}, got {actual}")
    return actual


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def evidence_record(evidence_id: str, path: Path, *, status: str, claim_ids: list[str]) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "artifact_path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "status": status,
        "claim_ids": claim_ids,
        "commit": git_commit(),
        "protocol_sha256": verify_protocol(),
    }


def require_file(path: Path, reason: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{reason}: missing {path.relative_to(ROOT)}")
