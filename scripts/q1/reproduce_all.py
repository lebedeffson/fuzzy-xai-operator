#!/usr/bin/env python3
"""One-command Q1 remediation orchestrator."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

from fuzzyxai.q1_validation.protocols import run_controlled_q1, write_json


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "release_evidence/q1_remediation"


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def command(*parts: str) -> None:
    print("RUN:", " ".join(parts), flush=True)
    subprocess.run(
        parts,
        cwd=ROOT,
        check=True,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    )


def main(profile: str) -> None:
    n_objects = 1_200 if profile == "smoke" else 10_000
    command(sys.executable, "scripts/q1/build_baseline_snapshot.py")
    run_controlled_q1(OUTPUT, n_objects=n_objects)
    write_json(
        OUTPUT / "run_manifest.json",
        {
            "schema_version": "1.0",
            "profile": profile,
            "base_commit": "cafe403c7d60e36b08f56a5325ba380718a5be35",
            "commit": os.environ.get("FUZZYXAI_COMMIT", git_value("rev-parse", "HEAD")),
            "branch": os.environ.get("FUZZYXAI_BRANCH", git_value("branch", "--show-current")),
            "python": platform.python_version(),
            "threads": 1,
            "controlled_objects": n_objects,
            "real_benchmark_status": "pending_heavy_ci" if profile == "smoke" else "required_separate_jobs",
            "external_gates": {
                "comprehension": "planned_not_run",
                "expert_action_review": "planned_not_run",
                "domain_language_review": "pending_external_review",
            },
            "stable_release_allowed": False,
        },
    )
    if profile == "full":
        real_dir = OUTPUT / "real_benchmarks"
        cache = ROOT / ".cache/q1"
        for modality in ("tabular", "image", "text", "timeseries"):
            command(
                sys.executable,
                "scripts/q1/run_real_benchmark.py",
                "--modality",
                modality,
                "--output",
                str(real_dir / f"{modality}.json"),
                "--cache",
                str(cache / modality),
            )
        command(sys.executable, "scripts/q1/merge_real_benchmarks.py", "--input-dir", str(real_dir))
    command(sys.executable, "scripts/q1/build_claim_registry.py")
    command(sys.executable, "scripts/q1/run_additional_protocols.py", "--profile", profile)
    command(sys.executable, "scripts/q1/build_reports.py")
    command(sys.executable, "scripts/q1/build_tables.py")
    command(sys.executable, "scripts/q1/build_figures.py")
    command(sys.executable, "scripts/q1/build_dod.py")
    command(sys.executable, "scripts/q1/build_manifest.py")
    command(sys.executable, "scripts/q1/verify_all.py")
    print(f"PASS: q1_controlled_protocol profile={profile} objects={n_objects}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    arguments = parser.parse_args()
    sys.exit(main(arguments.profile))
