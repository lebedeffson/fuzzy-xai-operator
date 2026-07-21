#!/usr/bin/env python3
"""Aggregate final H1-H5 real-artifact replications."""

from __future__ import annotations

import argparse
from pathlib import Path

from fuzzyxai.q1_final.hypotheses import run_hypotheses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("release_evidence/q1_final/hypotheses/final_results.json"),
    )
    args = parser.parse_args()
    result = run_hypotheses(args.input_dir, args.output)
    statuses = {
        "H1": result["H1_real"]["status"],
        "H2": result["H2_real"]["status"],
        "H3_full": result["H3_real"]["full_population_status"],
        "H3_hard": result["H3_real"]["hard_case_status"],
        "H4": result["H4_real"]["status"],
        "H5_structural": result["H5_real"]["structural"]["status"],
        "H5_predictive": result["H5_real"]["predictive"]["status"],
    }
    print(f"PASS: q1_final_hypotheses statuses={statuses}")


if __name__ == "__main__":
    main()
