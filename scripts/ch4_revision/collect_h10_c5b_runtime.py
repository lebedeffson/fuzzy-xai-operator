#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

TRACE_PATTERNS = ('File "', ": in ")


def _has_trace(value: str) -> bool:
    return any(pattern in value for pattern in TRACE_PATTERNS)


def collect(
    manifest_path: Path,
    command_registry_path: Path,
    output: Path,
) -> dict[str, object]:
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    registry = json.loads(command_registry_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    enriched = []
    evidence_rows = []
    for row in rows:
        incident_id = str(row["incident_id"])
        registered = registry.get(incident_id)
        if not registered:
            enriched.append(
                {**row, "runtime_evidence_status": "RUNTIME_COMMAND_NOT_REGISTERED"}
            )
            evidence_rows.append(
                {
                    "incident_id": incident_id,
                    "status": "RUNTIME_COMMAND_NOT_REGISTERED",
                }
            )
            continue
        command = tuple(str(item) for item in registered["command"])
        with tempfile.TemporaryDirectory(prefix="fuzzyxai-h10-c5b-runtime-") as temporary:
            sandbox = Path(temporary) / "repository"
            shutil.copytree(
                Path(str(row["repository_root"])),
                sandbox,
                ignore=shutil.ignore_patterns(
                    ".git",
                    ".tox",
                    ".venv",
                    "__pycache__",
                    ".pytest_cache",
                ),
            )
            completed = subprocess.run(
                command,
                cwd=sandbox,
                capture_output=True,
                text=True,
                timeout=int(registered.get("timeout_seconds", 900)),
                check=False,
                env=None,
            )
        combined = f"{completed.stdout}\n{completed.stderr}"
        reproduced = completed.returncode != 0
        status = (
            "BUG_REPRODUCED_WITH_TRACE"
            if reproduced and _has_trace(combined)
            else (
                "BUG_REPRODUCED_WITHOUT_TRACE"
                if reproduced
                else "BUG_NOT_REPRODUCED"
            )
        )
        incident_output = output / incident_id
        incident_output.mkdir(parents=True, exist_ok=True)
        stdout_path = incident_output / "stdout.txt"
        stderr_path = incident_output / "stderr.txt"
        traceback_path = incident_output / "traceback.txt"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        traceback_path.write_text(combined if _has_trace(combined) else "", encoding="utf-8")
        enriched.append(
            {
                **row,
                "stdout_path": str(stdout_path.resolve()),
                "stderr_path": str(stderr_path.resolve()),
                "traceback_path": str(traceback_path.resolve()),
                "runtime_evidence_status": status,
            }
        )
        evidence_rows.append(
            {
                "incident_id": incident_id,
                "status": status,
                "returncode": completed.returncode,
                "command_sha256": hashlib.sha256(
                    json.dumps(command, separators=(",", ":")).encode()
                ).hexdigest(),
                "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
            }
        )
    enriched_path = output / "H10_C5B_RUNTIME_ENRICHED_MANIFEST.jsonl"
    enriched_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in enriched),
        encoding="utf-8",
    )
    report = {
        "status": (
            "PASS"
            if evidence_rows
            and all(row["status"] == "BUG_REPRODUCED_WITH_TRACE" for row in evidence_rows)
            else "INCOMPLETE"
        ),
        "incident_count": len(rows),
        "trace_complete_count": sum(
            row["status"] == "BUG_REPRODUCED_WITH_TRACE" for row in evidence_rows
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "manifest": str(enriched_path),
        "evidence": evidence_rows,
    }
    (output / "RUNTIME_EVIDENCE_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--commands", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = collect(
        args.manifest.resolve(),
        args.commands.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
