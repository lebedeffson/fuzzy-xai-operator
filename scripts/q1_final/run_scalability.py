#!/usr/bin/env python3
"""Run final end-to-end scalability measurements."""

from __future__ import annotations

import argparse
from pathlib import Path

from fuzzyxai.q1_final.scalability import run_end_to_end_scalability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("release_evidence/q1_final/scalability/end_to_end.json"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    sizes = (100, 500) if args.smoke else (1_000, 5_000, 10_000, 50_000, 100_000)
    result = run_end_to_end_scalability(args.output, sizes=sizes)
    print(f"PASS: q1_final_scalability sizes={len(result['measurements'])}")


if __name__ == "__main__":
    main()
