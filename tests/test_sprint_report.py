from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sprint_report_smoke() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_sprint_report.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    out = ROOT / "reports" / "release" / "current"
    assert (out / "SPRINT_STATUS.md").exists()
    assert (out / "release_summary.json").exists()
    assert (out / "scenario_matrix.json").exists()

    summary = json.loads((out / "release_summary.json").read_text(encoding="utf-8"))
    matrix = json.loads((out / "scenario_matrix.json").read_text(encoding="utf-8"))
    assert summary["sprint_report_status"] == "PASS"
    assert summary["scenarios_total"] == 5
    assert summary["scenarios_passed"] == 5
    assert len(matrix) == 5
