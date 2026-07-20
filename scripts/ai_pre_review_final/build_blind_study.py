#!/usr/bin/env python3
"""Build final leakage-free reviewer records, batches and private scoring key."""

from __future__ import annotations

import json
from pathlib import Path

from fuzzyxai.ai_pre_review_final import build_final_blind_study


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    result = build_final_blind_study(ROOT)
    print(json.dumps({"status": "PASS", "cases": result["reviewer_cases"], "variants": result["reviewer_variants"], "batches": len(result["batches"])}, sort_keys=True))
