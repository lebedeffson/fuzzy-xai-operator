from __future__ import annotations

from pathlib import Path

import pytest

from h10_c2.hashing import write_json
from h10_c2.sealing.open_guard import validate_approval
from h10_c2.sealing.scoring_gate import freeze_protocol


def test_invalid_approval_variants(tmp_path: Path) -> None:
    path = tmp_path / "approval.json"
    write_json(
        path,
        {
            "approved": False,
            "approved_by": "owner",
            "signature": "sig",
            "protocol_sha256": "a" * 64,
            "purpose": "scoring_only",
        },
    )
    with pytest.raises(PermissionError, match="INCOMPLETE"):
        validate_approval(path, "a" * 64)
    write_json(
        path,
        {
            "approved": True,
            "approved_by": "owner",
            "signature": "sig",
            "protocol_sha256": "a" * 64,
            "purpose": "development",
        },
    )
    with pytest.raises(PermissionError, match="INVALID"):
        validate_approval(path, "a" * 64)


def test_protocol_freeze_rejects_unsigned_design(tmp_path: Path) -> None:
    path = tmp_path / "approval.json"
    write_json(path, {"approved": False, "recommended_design_sha256": ""})
    with pytest.raises(PermissionError, match="protocol-owner"):
        freeze_protocol(path)
