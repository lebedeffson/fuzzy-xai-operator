#!/usr/bin/env python3
"""Compare frozen AI scores with independent human consensus."""

from __future__ import annotations

import json
from pathlib import Path

from fuzzyxai.ai_pre_review import AI_RUN_IDS, compare_ai_human
from fuzzyxai.ai_pre_review.contracts import read_jsonl

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    base = ROOT / "release_evidence/ai_pre_review"
    ai_rows = [row for run in AI_RUN_IDS for row in read_jsonl(base / f"raw_ai/confirmatory/{run}.jsonl")]
    human_rows = read_jsonl(base / "raw_human/human_reviews.jsonl")
    config = json.loads((ROOT / "configs/ai_pre_review/config.json").read_text(encoding="utf-8"))
    result = compare_ai_human(ai_rows, human_rows, config["agreement_thresholds"])
    output = base / "ai_human_agreement.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: ai_human_comparison gate={str(result['threshold_gate_passed']).lower()}")
