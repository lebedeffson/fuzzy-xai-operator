from __future__ import annotations

import sys
from pathlib import Path

from fuzzyxai.repository_diagnostics import (
    IncidentSandboxExecutor,
    RegisteredRepair,
)


def test_expressible_repair_runs_failure_regression_and_recertification(tmp_path: Path) -> None:
    (tmp_path / "value.txt").write_text("broken\n", encoding="utf-8")
    (tmp_path / "test_bug.py").write_text(
        "from pathlib import Path\n"
        "raise SystemExit(0 if Path('value.txt').read_text().strip() == 'fixed' else 1)\n",
        encoding="utf-8",
    )
    (tmp_path / "test_regression.py").write_text("raise SystemExit(0)\n", encoding="utf-8")

    def repair(root: Path) -> None:
        (root / "value.txt").write_text("fixed\n", encoding="utf-8")

    report = IncidentSandboxExecutor().execute(
        tmp_path,
        RegisteredRepair("repair-value", "CONFIGURATION", "value.txt", repair),
        fail_to_pass_command=(sys.executable, "test_bug.py"),
        regression_command=(sys.executable, "test_regression.py"),
        recertify=lambda root: (root / "value.txt").read_text().strip() == "fixed",
    )
    assert report.statuses == (
        "BUG_REPRODUCED",
        "REPAIR_EXECUTED",
        "FAIL_TO_PASS_PASS",
        "REGRESSION_PASS",
        "RECERTIFICATION_PASS",
    )
    assert report.recertification_success
    assert report.before_sha256 != report.after_sha256
    assert (tmp_path / "value.txt").read_text().strip() == "broken"
