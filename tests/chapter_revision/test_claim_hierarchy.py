from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_h10_c3_hierarchy_preserves_sealed_values() -> None:
    subprocess.run(
        [sys.executable, "scripts/ch4_revision/build_h10_c3_hierarchy.py"],
        check=True,
    )
    payload = json.loads(Path("reports/chapter_revision/H10_C3_STATISTICAL_HIERARCHY.json").read_text())
    assert payload["primary_endpoint"] == "H10-C3a"
    assert payload["linked_secondary_endpoint"] == "H10-C3b"
    assert payload["independent_replication_claim"] is False


def test_claim_lint_rejects_removed_object_test(tmp_path: Path) -> None:
    bad = tmp_path / "bad.md"
    bad.write_text("p = 0.0234", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/ch4_revision/claim_lint.py",
            "--root",
            ".",
            str(bad),
        ],
        check=False,
    )
    assert result.returncode == 1
