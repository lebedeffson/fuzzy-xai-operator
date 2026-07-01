#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "external_validation" / "outputs"


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
    route_path = OUTPUTS / "external_wine_route.json"
    proof_path = OUTPUTS / "external_wine_proof_trace.json"
    dashboard_path = OUTPUTS / "external_wine_operator_dashboard.png"
    for path in (summary_path, route_path, proof_path, dashboard_path):
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

    print(f"import path: {import_path}")
    print(f"external task: {summary['task']}")
    print(
        "external result: "
        f"action={summary['action']}; "
        f"diagnostic={summary['diagnostic']}; "
        f"gamma={summary['computed_result'].get('gamma')}; "
        f"delta={summary['computed_result'].get('delta')}; "
        f"rho={summary['computed_result'].get('rho')}; "
        f"verifier={summary['verifier']}"
    )
    print("framework-external-check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
