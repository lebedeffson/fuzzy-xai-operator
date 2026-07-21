#!/usr/bin/env python3
"""Validate and freeze externally produced AI review JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path

from fuzzyxai.ai_pre_review import validate_ai_review_directory
from fuzzyxai.ai_pre_review.review_io import write_review_import


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("formative", "confirmatory"), required=True)
    parser.add_argument("--run-id", choices=("AI_RUN_1", "AI_RUN_2", "AI_RUN_3"), required=True)
    args = parser.parse_args()
    rows = validate_ai_review_directory(ROOT, args.review_dir, split=args.split, run_id=args.run_id)
    output = ROOT / f"release_evidence/ai_pre_review/raw_ai/{args.split}/{args.run_id}.jsonl"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite frozen AI reviews: {output}")
    write_review_import(output, rows)
    print(f"PASS: imported_ai_reviews split={args.split} run={args.run_id} rows={len(rows)}")


if __name__ == "__main__":
    main()
