from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.ch4_revision.collect_h10_c5b_runtime import collect


def test_runtime_collection_requires_reproduced_trace(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "failure.py").write_text(
        "raise ValueError('registered failure')\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "incident_id": "incident-1",
                "repository": "fixture/repo",
                "repository_root": str(repository),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    commands = tmp_path / "commands.json"
    commands.write_text(
        json.dumps(
            {
                "incident-1": {
                    "command": [sys.executable, "failure.py"],
                    "timeout_seconds": 30,
                }
            }
        ),
        encoding="utf-8",
    )
    report = collect(manifest, commands, tmp_path / "evidence")
    assert report["status"] == "PASS"
    assert report["trace_complete_count"] == 1
    enriched = json.loads(
        (tmp_path / "evidence/H10_C5B_RUNTIME_ENRICHED_MANIFEST.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert enriched["runtime_evidence_status"] == "BUG_REPRODUCED_WITH_TRACE"


def test_unregistered_runtime_command_is_incomplete(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "incident_id": "incident-1",
                "repository": "fixture/repo",
                "repository_root": str(tmp_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    commands = tmp_path / "commands.json"
    commands.write_text("{}", encoding="utf-8")
    report = collect(manifest, commands, tmp_path / "evidence")
    assert report["status"] == "INCOMPLETE"
    assert report["trace_complete_count"] == 0
