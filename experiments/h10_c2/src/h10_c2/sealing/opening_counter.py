from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..hashing import read_json, write_json


def initialize(path: Path) -> dict:
    if path.exists():
        return read_json(path)
    value = {"opening_count": 0, "events": []}
    write_json(path, value)
    return value


def record_opening(path: Path, event: dict, *, limit: int = 1) -> dict:
    value = initialize(path)
    if int(value["opening_count"]) >= limit:
        raise PermissionError("SEALED_REOPENING_FORBIDDEN")
    event = {**event, "opened_at_utc": datetime.now(timezone.utc).isoformat()}
    updated = {"opening_count": int(value["opening_count"]) + 1, "events": [*value["events"], event]}
    write_json(path, updated)
    return updated

