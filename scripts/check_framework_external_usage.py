#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "external_validation" / "outputs"
MODEL_KEYS = ("logistic_regression", "gradient_boosting")
ZIP_PATH = OUTPUTS / "external_wine_blackbox_validation.zip"


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="fuzzyxai-external-") as tmp:
        tmp_path = Path(tmp)
        probe = run(
            [
                sys.executable,
                "-c",
                "import fuzzyxai; print(fuzzyxai.__file__)",
            ],
            tmp_path,
        )
        if probe.returncode != 0:
            print(probe.stdout, end="")
            print(probe.stderr, end="", file=sys.stderr)
            return probe.returncode
        import_path = probe.stdout.strip()
        expected = (ROOT / "framework" / "fuzzyxai" / "fuzzyxai").as_posix()
        legacy = (ROOT / "fuzzyxai").as_posix()
        if expected not in import_path:
            print(f"framework-external-check: FAIL import path is not framework package: {import_path}", file=sys.stderr)
            return 1
        if import_path.startswith(legacy):
            print(f"framework-external-check: FAIL imported legacy root package: {import_path}", file=sys.stderr)
            return 1

        script = ROOT / "external_validation" / "run_external_wine_test.py"
        result = run([sys.executable, str(script)], tmp_path)
        print(result.stdout, end="")
        if result.returncode != 0:
            print(result.stderr, end="", file=sys.stderr)
            return result.returncode

    summary_path = OUTPUTS / "external_wine_summary.json"
    required = [summary_path, ZIP_PATH]
    for model_key in MODEL_KEYS:
        required.extend(
            [
                OUTPUTS / f"external_wine_{model_key}_route.json",
                OUTPUTS / f"external_wine_{model_key}_proof_trace.json",
                OUTPUTS / f"external_wine_{model_key}_operator_dashboard.png",
                OUTPUTS / f"external_wine_{model_key}_summary.json",
            ]
        )
    for path in required:
        if not path.exists() or path.stat().st_size == 0:
            print(f"framework-external-check: FAIL missing output {path}", file=sys.stderr)
            return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("verifier") != "passed":
        print("framework-external-check: FAIL verifier is not passed", file=sys.stderr)
        return 1
    if not summary.get("source_commit"):
        print("framework-external-check: FAIL source_commit is empty", file=sys.stderr)
        return 1
    validations = summary.get("validations") or []
    if len(validations) != 2:
        print(f"framework-external-check: FAIL expected 2 external validations, got {len(validations)}", file=sys.stderr)
        return 1
    for item in validations:
        computed = item.get("computed_result") or {}
        if item.get("action") != "lower_confidence":
            print(f"framework-external-check: FAIL {item.get('model_key')} action is {item.get('action')}", file=sys.stderr)
            return 1
        if item.get("diagnostic") != "D_external_tabular_uncertainty":
            print(f"framework-external-check: FAIL {item.get('model_key')} diagnostic is {item.get('diagnostic')}", file=sys.stderr)
            return 1
        for key in ("gamma", "delta", "rho"):
            value = float(computed.get(key, 0.0))
            if value <= 0.0:
                print(f"framework-external-check: FAIL {item.get('model_key')} {key} is zero", file=sys.stderr)
                return 1
        gamma = float(computed["gamma"])
        delta = float(computed["delta"])
        rho = float(computed["rho"])
        if not (0.10 <= gamma <= 0.60 and 0.05 <= delta <= 0.60 and 0.10 <= rho <= 0.70):
            print(
                f"framework-external-check: FAIL {item.get('model_key')} values out of range: "
                f"gamma={gamma}, delta={delta}, rho={rho}",
                file=sys.stderr,
            )
            return 1

    print(f"import path: {import_path}")
    print(f"external task: {summary['task']}")
    for item in validations:
        computed = item["computed_result"]
        print(
            "external result: "
            f"model={item['model_name']}; "
            f"action={item['action']}; "
            f"diagnostic={item['diagnostic']}; "
            f"gamma={computed.get('gamma')}; "
            f"delta={computed.get('delta')}; "
            f"rho={computed.get('rho')}; "
            f"verifier={item['verifier']}"
        )
    print("framework-external-check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
