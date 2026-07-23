from __future__ import annotations

from pathlib import Path

import pytest

from h10_c2.hashing import write_json
from h10_c2.sealing.open_guard import validate_approval
from h10_c2.sealing.opening_counter import initialize, record_opening


def test_opening_counter_starts_closed(tmp_path: Path) -> None:
    path = tmp_path / "opening.json"
    assert initialize(path) == {"opening_count": 0, "events": []}


def test_opening_counter_rejects_second_opening(tmp_path: Path) -> None:
    path = tmp_path / "opening.json"
    record_opening(path, {"purpose": "scoring_only"})
    with pytest.raises(PermissionError, match="REOPENING"):
        record_opening(path, {"purpose": "scoring_only"})


def test_approval_requires_matching_protocol(tmp_path: Path) -> None:
    path = tmp_path / "approval.json"
    write_json(
        path,
        {
            "approved": True,
            "approved_by": "review-board",
            "signature": "external-signature",
            "protocol_sha256": "a" * 64,
            "purpose": "scoring_only",
        },
    )
    assert validate_approval(path, "a" * 64)["approved"]
    with pytest.raises(PermissionError, match="MISMATCH"):
        validate_approval(path, "b" * 64)


def test_incomplete_approval_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "approval.json"
    write_json(path, {"approved": True})
    with pytest.raises(PermissionError, match="INCOMPLETE"):
        validate_approval(path, "a" * 64)

