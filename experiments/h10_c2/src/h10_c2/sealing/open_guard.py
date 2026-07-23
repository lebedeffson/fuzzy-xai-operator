from __future__ import annotations

from pathlib import Path

from ..hashing import read_json


def validate_approval(path: Path, expected_protocol_sha256: str) -> dict:
    if not path.exists():
        raise PermissionError("SEALED_APPROVAL_REQUIRED")
    value = read_json(path)
    required = ("approved", "approved_by", "signature", "protocol_sha256", "purpose")
    if any(not value.get(key) for key in required):
        raise PermissionError("SEALED_APPROVAL_INCOMPLETE")
    if value["approved"] is not True or value["purpose"] != "scoring_only":
        raise PermissionError("SEALED_APPROVAL_INVALID")
    if value["protocol_sha256"] != expected_protocol_sha256:
        raise PermissionError("SEALED_APPROVAL_PROTOCOL_MISMATCH")
    return value

