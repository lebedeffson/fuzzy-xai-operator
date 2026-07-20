#!/usr/bin/env python3
"""Run the preregistered H6 confirmatory protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from fuzzyxai.q1_final.rule_ablation import run_confirmatory_rule_ablation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("release_evidence/q1_final/rule_ablation"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/q1-final/rule-ablation"))
    parser.add_argument("--folds", type=int, default=10)
    args = parser.parse_args()
    result = run_confirmatory_rule_ablation(args.output, args.cache, folds=args.folds)
    print(f"PASS: q1_final_h6 status={result['status']} comparisons={result['n_comparisons']}")


if __name__ == "__main__":
    main()
