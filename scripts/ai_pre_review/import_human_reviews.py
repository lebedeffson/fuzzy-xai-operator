#!/usr/bin/env python3
"""Import externally authored and signed human review records."""

from __future__ import annotations

import argparse
from pathlib import Path

from fuzzyxai.ai_pre_review import validate_human_review_directory
from fuzzyxai.ai_pre_review.review_io import write_review_import

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = validate_human_review_directory(ROOT, args.review_dir)
    output = ROOT / "release_evidence/ai_pre_review/raw_human/human_reviews.jsonl"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite frozen human reviews: {output}")
    write_review_import(output, rows)
    print(f"PASS: imported_human_reviews rows={len(rows)}")
