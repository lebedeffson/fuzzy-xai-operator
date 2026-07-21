#!/usr/bin/env python3
"""Run the blocking blindness and explanation-quality audit."""

from __future__ import annotations

import json
from pathlib import Path

from fuzzyxai.ai_pre_review_final import audit_blind_records
from fuzzyxai.ai_pre_review_final.contracts import read_jsonl


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    public_dir = ROOT / "study/ai_pre_review_final/public_formative"
    rows = read_jsonl(public_dir / "reviewer_cases.jsonl")
    result = audit_blind_records(rows, root=public_dir, expected_cases=240, expected_records=720)
    output = ROOT / "reports/ai_pre_review_final/BLINDING_AUDIT.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: final_blinding_audit records={result['records']} cases={result['cases']} coverage={result['claim_evidence_coverage_min']:.1f}")
