from __future__ import annotations

import json
import subprocess
import sys
import zipfile
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
    package_dir = out / "external_wine_blackbox_validation"
    summary = json.loads((out / "external_wine_summary.json").read_text(encoding="utf-8"))

    assert summary["verifier"] == "passed"
    assert (out / "external_wine_blackbox_validation.zip").stat().st_size > 0
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads((package_dir / "import_provenance.json").read_text(encoding="utf-8"))
    assert manifest["source_commit"] == summary["source_commit"]
    assert provenance["package_boundary_ok"] is True
    assert provenance["applications_used"] is False
    assert str(ROOT) not in json.dumps(summary)
    with zipfile.ZipFile(out / "external_wine_blackbox_validation.zip") as archive:
        names = set(archive.namelist())
    assert "external_wine_blackbox_validation/manifest.json" in names
    assert "external_wine_blackbox_validation/logistic_regression/route.json" in names
    assert "external_wine_blackbox_validation/gradient_boosting/route.json" in names
    assert len(summary["validations"]) == 2
    for item in summary["validations"]:
        model_key = item["model_key"]
        route = json.loads((out / f"external_wine_{model_key}_route.json").read_text(encoding="utf-8"))
        proof = json.loads((out / f"external_wine_{model_key}_proof_trace.json").read_text(encoding="utf-8"))
        computed = item["computed_result"]
        assert (out / f"external_wine_{model_key}_operator_dashboard.png").stat().st_size > 0
        assert item["verifier"] == "passed"
        assert item["action"] == "lower_confidence"
        assert item["diagnostic"] == "D_external_tabular_uncertainty"
        assert 0.10 <= computed["gamma"] <= 0.60
        assert 0.05 <= computed["delta"] <= 0.60
        assert 0.10 <= computed["rho"] <= 0.70
        assert route["source_commit"]
        assert proof["source_commit"]
        assert item["route"] == f"{model_key}/route.json"
        assert item["proof_trace"] == f"{model_key}/proof_trace.json"
        assert item["dashboard"] == f"{model_key}/operator_dashboard.png"
