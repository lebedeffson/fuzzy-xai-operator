#!/usr/bin/env python3
"""Run required explainers on one modality's frozen cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

from fuzzyxai.q1_final.explainers import run_explainer_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=("tabular", "image", "text", "timeseries"), required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--neural", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    payload = run_explainer_evaluation(
        args.modality,
        args.benchmark,
        args.output,
        args.cache,
        neural_path=args.neural,
    )
    print(f"PASS: q1_final_explainers modality={args.modality} methods={len(payload['methods'])}")


if __name__ == "__main__":
    main()
