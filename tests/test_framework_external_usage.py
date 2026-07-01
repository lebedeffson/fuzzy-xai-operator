from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_framework_external_usage_from_tmp(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "framework/fuzzyxai"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    probe = subprocess.run(
        [sys.executable, "-c", "import fuzzyxai; print(fuzzyxai.__file__)"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    import_path = probe.stdout.strip()
    assert "framework/fuzzyxai/fuzzyxai/__init__.py" in import_path
    assert "/fuzzyxai-operator/fuzzyxai/__init__.py" not in import_path

    result = subprocess.run(
        [sys.executable, str(ROOT / "external_validation" / "run_external_wine_test.py")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    out = ROOT / "external_validation" / "outputs"
    route = json.loads((out / "external_wine_route.json").read_text(encoding="utf-8"))
    proof = json.loads((out / "external_wine_proof_trace.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "external_wine_summary.json").read_text(encoding="utf-8"))

    assert (out / "external_wine_operator_dashboard.png").stat().st_size > 0
    assert summary["verifier"] == "passed"
    assert summary["action"] == "accept"
    assert summary["diagnostic"] == "D_external_tabular_ok"
    assert route["source_commit"]
    assert proof["source_commit"]
