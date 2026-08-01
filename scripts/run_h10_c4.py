#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from fuzzyxai.experiments.h10_c4 import run_experiment, verify_outputs

ROOT = Path(__file__).resolve().parents[1]


def _verify_immutability() -> dict[str, object]:
    completed = subprocess.run(
        [
            "sha256sum",
            "-c",
            "protocol/h10_c4/H10_C3_BASELINE_SHA256SUMS",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "checked_files": sum(
            line.endswith(": OK") for line in completed.stdout.splitlines()
        ),
        "stderr": completed.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prospective H10-C4")
    parser.add_argument("command", choices=("run", "verify"))
    args = parser.parse_args()

    if args.command == "run":
        print(json.dumps(run_experiment(ROOT), indent=2, sort_keys=True))
        return 0

    immutability = _verify_immutability()
    outputs = verify_outputs(ROOT)
    report = {
        "h10_c3_immutability": immutability,
        "h10_c4_outputs": outputs,
        "status": (
            "PASS"
            if immutability["status"] == "PASS" and outputs["status"] == "PASS"
            else "FAIL"
        ),
    }
    output = ROOT / "reports/h10_c4/verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
