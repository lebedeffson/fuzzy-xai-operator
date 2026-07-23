from __future__ import annotations

from ..hashing import read_json
from ..paths import ARTIFACT_ROOT


def adjudication_status() -> str:
    path = ARTIFACT_ROOT / "adjudication" / "status.json"
    if not path.exists():
        return "BLOCKED_HUMAN_ADJUDICATION"
    return str(read_json(path).get("status", "BLOCKED_HUMAN_ADJUDICATION"))

