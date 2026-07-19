#!/usr/bin/env python3
"""One-command dissertation reproduction orchestrator."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command(*parts: str) -> None:
    print("RUN:", " ".join(parts), flush=True)
    subprocess.run(parts, cwd=ROOT, check=True, env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})


def run(profile: str, *, skip_optional: bool, skip_archives: bool) -> None:
    python = sys.executable
    command(python, "scripts/run_full_empirical_validation.py", "--profile", profile)
    if not skip_optional:
        objects = "1000" if profile == "smoke" else "10000"
        command(python, "scripts/run_optional_multimodal_models.py", "--objects", objects)
    command(python, "scripts/rebuild_empirical_manifest.py")
    command(python, "scripts/build_dissertation_tables.py")
    command(python, "scripts/build_dissertation_figures.py")
    command(python, "scripts/build_dissertation_claims.py")
    command(python, "scripts/build_full_empirical_report.py")
    command(python, "scripts/build_empirical_dod.py")
    command(python, "scripts/rebuild_empirical_manifest.py")
    command(python, "scripts/verify_reproduction.py", "--profile", profile)
    if not skip_archives:
        command(python, "scripts/build_empirical_archives.py")
    print("PASS: reproduce_dissertation_complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--skip-optional", action="store_true")
    parser.add_argument("--skip-archives", action="store_true")
    args = parser.parse_args()
    run(args.profile, skip_optional=args.skip_optional, skip_archives=args.skip_archives)
