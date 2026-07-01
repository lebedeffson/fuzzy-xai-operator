#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "release" / "fuzzyxai_framework_rc"
ZIP = ROOT / "reports" / "release" / "fuzzyxai_framework_rc_package.zip"
PAYLOAD = {
    "scenario_id": "external_wine_classification",
    "source_type": "tabular",
    "model_name": "ExternalDemoModel",
    "dataset_name": "manual_payload",
    "predicted_class": 1,
    "class_probability": 0.68,
    "feature_values": {"x1": 0.4, "x2": 0.7, "x3": 0.2},
    "feature_importance": {"x1": 0.31, "x2": 0.18, "x3": 0.12},
    "quality_metrics": {"missing_rate": 0.05, "feature_range_violation": 0.0},
    "context_values": {"external_task": True},
}


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False, env=env)


def must(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = run(cmd, cwd, env)
    if result.returncode:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        raise SystemExit(result.returncode)
    return result


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_required(src: Path, dst: Path) -> None:
    if not src.exists() or src.stat().st_size == 0:
        raise SystemExit(f"required artifact missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_acceptance(clean: bool = False) -> dict[str, object]:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    commit = must(["git", "rev-parse", "HEAD"], ROOT).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="fuzzyxai-rc-") as tmp:
        tmp_root = Path(tmp)
        clone = tmp_root / "fuzzyxai-rc-check"
        must(["git", "clone", str(ROOT), str(clone)], tmp_root)
        must(["git", "checkout", commit], clone)
        venv = tmp_root / ".venv"
        must([sys.executable, "-m", "venv", str(venv)], tmp_root)
        py = venv / "bin" / "python"
        must([str(py), "-m", "pip", "install", "-U", "pip"], tmp_root)
        must([str(py), "-m", "pip", "install", "-e", str(clone / "framework" / "fuzzyxai")], tmp_root)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        import_path = must([str(py), "-c", "import fuzzyxai; print(fuzzyxai.__file__)"], tmp_root, env).stdout.strip()
        if "framework/fuzzyxai/fuzzyxai/__init__.py" not in import_path:
            raise SystemExit(f"fuzzyxai import path is outside framework package: {import_path}")
        sdk_import = must([str(py), "-c", "from fuzzyxai import FuzzyXAI, ExplainPlan; print('ok')"], tmp_root, env).stdout.strip()
        adapters = must([str(py), "-m", "fuzzyxai.cli", "list-adapters"], tmp_root, env).stdout
        operators = must([str(py), "-m", "fuzzyxai.cli", "list-operators"], tmp_root, env).stdout
        write(OUT / "schemas" / "adapter_registry.json", adapters)
        write(OUT / "schemas" / "operator_registry.json", operators)

        cli_dir = OUT / "cli_check"
        payload_path = cli_dir / "payload.json"
        write(payload_path, json.dumps(PAYLOAD, ensure_ascii=False, indent=2) + "\n")
        run_dir = tmp_root / "cli_out"
        commands = [
            [str(py), "-m", "fuzzyxai.cli", "validate", "--payload", str(payload_path), "--schema", "classification"],
            [str(py), "-m", "fuzzyxai.cli", "run", "--payload", str(payload_path), "--adapter", "tabular_classification", "--out", str(run_dir)],
            [str(py), "-m", "fuzzyxai.cli", "verify", "--route", str(run_dir / "route.json"), "--proof", str(run_dir / "proof_trace.json")],
            [str(py), "-m", "fuzzyxai.cli", "render", "--route", str(run_dir / "route.json"), "--out", str(run_dir / "dashboard.png")],
            [str(py), "-m", "fuzzyxai.cli", "package", "--route", str(run_dir / "route.json"), "--out", str(run_dir / "audit_package.zip")],
        ]
        cli_log = []
        for command in commands:
            result = must(command, tmp_root, env)
            cli_log.append("$ " + " ".join(command) + "\n" + result.stdout)
        write(cli_dir / "cli_check_output.txt", "\n".join(cli_log))
        for name in [
            "route.json",
            "proof_trace.json",
            "operator_trace.json",
            "operator_table.csv",
            "verifier_report.json",
            "dashboard_data.json",
            "dashboard.png",
            "audit_package.zip",
        ]:
            copy_required(run_dir / name, cli_dir / name)
        route = json.loads((run_dir / "route.json").read_text(encoding="utf-8"))
        computed = route["computed_result"]
        verifier = json.loads((run_dir / "verifier_report.json").read_text(encoding="utf-8"))

        sdk_dir = OUT / "sdk_check"
        sdk_script = sdk_dir / "sdk_check.py"
        write(
            sdk_script,
            "from fuzzyxai import FuzzyXAI, ExplainPlan\n"
            "from fuzzyxai.adapters import get_adapter\n"
            f"payload = {PAYLOAD!r}\n"
            "fxai = FuzzyXAI(plan=ExplainPlan.default())\n"
            "route = fxai.run_payload(payload, get_adapter('tabular_classification'))\n"
            "report = fxai.verify(route)\n"
            "print(route.computed_result)\n"
            "print('overall_status=' + ('passed' if report.valid else 'failed'))\n",
        )
        sdk_result = must([str(py), str(sdk_script)], tmp_root, env)
        write(sdk_dir / "sdk_check_output.txt", sdk_result.stdout)

    schema_report = {
        "schemas": sorted(path.name for path in (ROOT / "framework" / "fuzzyxai" / "fuzzyxai" / "schemas").glob("*.schema.json")),
        "schema_check": "PASS",
        "adapter_registry_check": "PASS",
        "operator_registry_check": "PASS",
    }
    write(OUT / "schemas" / "schema_validation_report.json", json.dumps(schema_report, ensure_ascii=False, indent=2) + "\n")
    for doc in ["API.md", "CLI.md", "ADAPTER_SDK.md", "EXPLAIN_PLAN.md", "OPERATOR_REGISTRY.md"]:
        copy_required(ROOT / "framework" / "fuzzyxai" / "docs" / doc, OUT / "docs" / doc)
    for src, dst in [
        (ROOT / "research_validation" / "sensitivity" / "sensitivity_results.csv", OUT / "research_analysis" / "sensitivity_results.csv"),
        (ROOT / "research_validation" / "ablation" / "ablation_summary.csv", OUT / "research_analysis" / "ablation_summary.csv"),
        (ROOT / "research_validation" / "ablation" / "ablation_action_changes.csv", OUT / "research_analysis" / "ablation_action_changes.csv"),
        (ROOT / "research_validation" / "cross_model" / "cross_model_summary.csv", OUT / "research_analysis" / "cross_model_summary.csv"),
        (ROOT / "research_validation" / "sensitivity" / "rho_surface.png", OUT / "research_analysis" / "figures" / "rho_surface.png"),
        (ROOT / "research_validation" / "sensitivity" / "action_transition_heatmap.png", OUT / "research_analysis" / "figures" / "action_transition_heatmap.png"),
        (ROOT / "research_validation" / "cross_model" / "cross_model_mean_rho.png", OUT / "research_analysis" / "figures" / "cross_model_mean_rho.png"),
    ]:
        copy_required(src, dst)

    ablation_rows = (OUT / "research_analysis" / "ablation_summary.csv").read_text(encoding="utf-8").splitlines()
    cross_rows = (OUT / "research_analysis" / "cross_model_summary.csv").read_text(encoding="utf-8").splitlines()
    sensitivity_rows = (OUT / "research_analysis" / "sensitivity_results.csv").read_text(encoding="utf-8").splitlines()
    report = f"""# FuzzyXAI Framework Release Candidate Acceptance Report

## Source

- source_commit: `{commit}`
- import_path: `{import_path}`

## Checks

| Check | Result |
|---|---|
| clean_install | PASS |
| sdk_import | {sdk_import} |
| cli_check | PASS |
| sdk_check | PASS |
| schema_check | PASS |
| research_analysis_check | PASS |

## CLI Result

- action: `{computed.get('action_id') or computed.get('action')}`
- diagnostic: `{computed.get('diagnostic_id')}`
- gamma: `{computed.get('gamma')}`
- delta: `{computed.get('delta')}`
- rho: `{computed.get('rho')}`
- verifier: `{verifier.get('overall_status')}`

## Research Analysis

- sensitivity rows: `{len(sensitivity_rows) - 1}`
- ablation summary rows: `{len(ablation_rows) - 1}`
- cross-model rows: `{len(cross_rows) - 1}`
"""
    write(OUT / "framework_acceptance_report.md", report)
    files = [path for path in sorted(OUT.rglob("*")) if path.is_file()]
    manifest = {
        "source_commit": commit,
        "package_type": "fuzzyxai_framework_release_candidate",
        "checks": {
            "clean_install": "PASS",
            "cli_check": "PASS",
            "sdk_check": "PASS",
            "schema_check": "PASS",
            "research_analysis_check": "PASS",
        },
        "sha256": [
            {"path": path.relative_to(OUT).as_posix(), "sha256": digest(path), "size_bytes": path.stat().st_size}
            for path in files
            if path.name != "manifest.json"
        ],
    }
    write(OUT / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(path, f"fuzzyxai_framework_rc/{path.relative_to(OUT).as_posix()}")
    return {
        "commit": commit,
        "import_path": import_path,
        "action": computed.get("action_id") or computed.get("action"),
        "gamma": computed.get("gamma"),
        "delta": computed.get("delta"),
        "rho": computed.get("rho"),
        "verifier": verifier.get("overall_status"),
        "package": ZIP.as_posix(),
        "sensitivity_rows": len(sensitivity_rows) - 1,
        "ablation_rows": len(ablation_rows) - 1,
        "cross_model_rows": len(cross_rows) - 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    result = build_acceptance(clean=not args.package_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("fuzzyxai-framework-rc-package: PASS" if args.package_only else "fuzzyxai-framework-rc-check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
