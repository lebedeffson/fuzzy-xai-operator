#!/usr/bin/env python3
"""Run one real Q1 modality benchmark for a heavy CI matrix job."""

from __future__ import annotations

import argparse
from pathlib import Path

from fuzzyxai.q1_validation.real_benchmarks import run_real_benchmark


ROOT = Path(__file__).resolve().parents[2]


def main(modality: str, output: Path, cache: Path) -> None:
    payload = run_real_benchmark(modality, output, cache)
    if payload["dataset"]["n_objects"] < 10_000:
        raise RuntimeError("real benchmark must contain at least 10,000 objects")
    failed = [row for row in payload["models"] + payload["explainers"] if row["status"] == "failed"]
    if failed:
        raise RuntimeError(f"failed required real benchmark channels: {failed}")
    print(f"PASS: q1_real_benchmark modality={modality} objects={payload['dataset']['n_objects']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=("tabular", "image", "text", "timeseries"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/q1")
    args = parser.parse_args()
    main(args.modality, args.output, args.cache)
