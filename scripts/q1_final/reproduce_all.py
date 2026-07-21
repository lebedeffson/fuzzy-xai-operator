#!/usr/bin/env python3
"""One-command orchestration for smoke or full final Q1 evidence."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "release_evidence/q1_final"
JOBS = EVIDENCE / "real_jobs"
ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join((str(ROOT / "framework/fuzzyxai"), str(ROOT))),
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def command(*parts: str) -> None:
    print("RUN:", " ".join(parts), flush=True)
    subprocess.run(parts, cwd=ROOT, env=ENV, check=True)


def main(profile: str, input_dir: Path | None) -> None:
    command(sys.executable, "scripts/q1_final/build_external_study_pack.py")
    command(sys.executable, "scripts/q1_final/verify_external_gates.py", "--allow-open")
    if profile == "full":
        JOBS.mkdir(parents=True, exist_ok=True)
        for modality in ("tabular", "image", "text", "timeseries"):
            command(
                sys.executable,
                "scripts/q1_final/run_multiclass_benchmark.py",
                "--modality",
                modality,
                "--output",
                str(JOBS / f"{modality}.json"),
                "--cache",
                str(ROOT / f".cache/q1-final/{modality}"),
            )
        for modality in ("image", "text", "timeseries"):
            command(
                sys.executable,
                "scripts/q1_final/run_neural_benchmark.py",
                "--modality",
                modality,
                "--output",
                str(JOBS / f"{modality}_neural.json"),
                "--cache",
                str(ROOT / f".cache/q1-final/{modality}"),
            )
        for modality in ("tabular", "image", "text", "timeseries"):
            parts = [
                sys.executable,
                "scripts/q1_final/run_explainers.py",
                "--modality",
                modality,
                "--benchmark",
                str(JOBS / f"{modality}.json"),
                "--output",
                str(JOBS / f"{modality}_explainers.json"),
                "--cache",
                str(ROOT / f".cache/q1-final/{modality}"),
            ]
            if modality == "image":
                parts.extend(("--neural", str(JOBS / "image_neural.json")))
            command(*parts)
        command(sys.executable, "scripts/q1_final/merge_real_benchmarks.py", "--input-dir", str(JOBS))
        command(sys.executable, "scripts/q1_final/run_hypotheses.py", "--input-dir", str(JOBS))
        command(sys.executable, "scripts/q1_final/run_rule_ablation.py")
        command(sys.executable, "scripts/q1_final/run_scalability.py")
    elif profile == "aggregate":
        if input_dir is None:
            raise ValueError("aggregate profile requires --input-dir")
        command(sys.executable, "scripts/q1_final/merge_real_benchmarks.py", "--input-dir", str(input_dir))
        command(sys.executable, "scripts/q1_final/run_hypotheses.py", "--input-dir", str(input_dir))
    else:
        command(sys.executable, "scripts/q1_final/run_scalability.py", "--smoke")
    command(sys.executable, "scripts/q1_final/build_identity.py")
    command(sys.executable, "scripts/q1_final/build_claim_registry.py")
    command(sys.executable, "scripts/q1_final/build_gate_matrix.py")
    command(sys.executable, "scripts/q1_final/build_reports.py")
    command(sys.executable, "scripts/q1_final/build_dod.py")
    command(sys.executable, "scripts/q1_final/build_manifest.py")
    command(sys.executable, "scripts/q1_final/check_forbidden_claims.py")
    command(sys.executable, "scripts/q1_final/verify_all.py", *(('--require-heavy',) if profile in {"full", "aggregate"} else ()))
    if profile == "full":
        command(sys.executable, "scripts/build_framework_release.py")
        command(sys.executable, "scripts/q1_final/build_archives.py")
        command(sys.executable, "scripts/q1_final/verify_archives.py")
        command(sys.executable, "scripts/q1_final/build_dod.py")
        command(sys.executable, "scripts/q1_final/build_manifest.py")
        # Rebuild once so every archive embeds the post-verification DoD and manifest.
        command(sys.executable, "scripts/q1_final/build_archives.py")
        command(sys.executable, "scripts/q1_final/verify_archives.py")
    print(f"PASS: q1_final_reproduction profile={profile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full", "aggregate"), default="smoke")
    parser.add_argument("--input-dir", type=Path)
    args = parser.parse_args()
    main(args.profile, args.input_dir)
