#!/usr/bin/env python3
"""Validate an external AI review directory without importing it."""

from __future__ import annotations

import argparse
from pathlib import Path

from fuzzyxai.ai_pre_review import validate_ai_review_directory


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("formative", "confirmatory"), required=True)
    parser.add_argument("--run-id", choices=("AI_RUN_1", "AI_RUN_2", "AI_RUN_3"), required=True)
    args = parser.parse_args()
    rows = validate_ai_review_directory(ROOT, args.review_dir, split=args.split, run_id=args.run_id)
    print(f"PASS: validated_ai_reviews rows={len(rows)}")
