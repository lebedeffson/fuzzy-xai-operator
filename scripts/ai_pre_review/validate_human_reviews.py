#!/usr/bin/env python3
"""Validate independently authored human responses without importing them."""

from __future__ import annotations

import argparse
from pathlib import Path

from fuzzyxai.ai_pre_review import validate_human_review_directory

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = validate_human_review_directory(ROOT, args.review_dir)
    print(f"PASS: validated_human_reviews rows={len(rows)}")
