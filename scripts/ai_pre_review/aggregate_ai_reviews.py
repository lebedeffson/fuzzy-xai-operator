#!/usr/bin/env python3
"""Aggregate three independently imported confirmatory AI runs."""

from __future__ import annotations

import json
from pathlib import Path

from fuzzyxai.ai_pre_review import AI_RUN_IDS, aggregate_ai_reviews


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    base = ROOT / "release_evidence/ai_pre_review/raw_ai/confirmatory"
    result = aggregate_ai_reviews(ROOT, {run: base / run for run in AI_RUN_IDS})
    output = ROOT / "release_evidence/ai_pre_review/ai_interrun_agreement.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: aggregate_ai_review kappa={result['weighted_kappa_mean']:.4f} icc={result['icc_total_score']:.4f}")
