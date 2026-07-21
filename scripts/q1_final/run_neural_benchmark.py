#!/usr/bin/env python3
"""Run one optional PyTorch native-modality benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from fuzzyxai.q1_final.neural import run_neural_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=("image", "text", "timeseries"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-cap", type=int, default=30_000)
    args = parser.parse_args()
    payload = run_neural_benchmark(
        args.modality,
        args.output,
        args.cache,
        epochs=args.epochs,
        train_cap=args.train_cap,
    )
    print(f"PASS: q1_final_neural modality={args.modality} runs={len(payload['models'])}")


if __name__ == "__main__":
    main()
