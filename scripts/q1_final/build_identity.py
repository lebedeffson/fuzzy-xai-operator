#!/usr/bin/env python3
"""Build one canonical runtime identity for all final Q1 artifacts."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

from fuzzyxai.q1_final import FinalRunIdentity


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "release_evidence/q1_final/run_identity.json"
BASE_COMMIT = "41c32af25242164144fd907e4850fa9d4f426bd1"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    final_commit = os.environ.get("FUZZYXAI_FINAL_COMMIT", git("rev-parse", "HEAD"))
    branch = os.environ.get("FUZZYXAI_FINAL_BRANCH", git("branch", "--show-current"))
    external_path = ROOT / "release_evidence/q1_final/external/status.json"
    external = json.loads(external_path.read_text(encoding="utf-8")) if external_path.is_file() else {"gates": {}}
    real_path = ROOT / "release_evidence/q1_final/real_benchmarks/combined_status.json"
    real = json.loads(real_path.read_text(encoding="utf-8")) if real_path.is_file() else {"status": "NOT_RUN"}
    run_ids = tuple(filter(None, os.environ.get("FUZZYXAI_CI_RUN_IDS", "").split(",")))
    identity = FinalRunIdentity(
        schema_version="2.0",
        branch=branch,
        base_commit=BASE_COMMIT,
        final_commit=final_commit,
        ci_run_ids=run_ids,
        profile="full_q1_final",
        real_benchmark_status=str(real["status"]).lower(),
        external_gate_status=dict(external.get("gates", {})),
        stable_release_allowed=bool(external.get("stable_release_allowed", False)) and str(real["status"]).upper() == "PASS",
        created_at=git("show", "-s", "--format=%cI", final_commit),
        python=platform.python_version(),
        platform=platform.platform(),
        threads=1,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(identity.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: q1_final_identity commit={final_commit} real={identity.real_benchmark_status}")


if __name__ == "__main__":
    main()
