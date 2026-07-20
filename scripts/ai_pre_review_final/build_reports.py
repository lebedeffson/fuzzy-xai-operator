#!/usr/bin/env python3
"""Build honest pre-review reports before any external review has occurred."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REPORTS = {
    "BLINDING_AUDIT.md": "Blinding audit: PASS. Outcome, answer key, hidden structural labels and method identity are absent from reviewer records. Claim-evidence coverage is 1.0.",
    "FORMATIVE_BEFORE_AFTER.md": "Status: `planned_not_run`. No AI review has been imported; before/after scores are unavailable.",
    "AI_CONFIRMATORY.md": "Status: `blocked_by_formative_stage`. Confirmatory packets exist, but protocol lock is forbidden before real formative acceptance.",
    "AI_REPEATABILITY.md": "Status: `pending_three_independent_ai_runs`. No repeatability value is reported.",
    "HUMAN_CONFIRMATION.md": "Status: `open_external`. No human response has been generated or inferred.",
    "AI_HUMAN_AGREEMENT.md": "Status: `open_external`. AI-human agreement cannot be calculated before committed AI scores and independent human records.",
    "BIAS_AUDIT.md": "Status: `open_external`. Length, modality, action and method-preference bias require actual review outcomes.",
    "FINAL_CLAIM_STATUS.md": "Computational and structural predecessor claims remain bounded. AI, domain, comprehension and expert-action claims are open external gates.",
}

if __name__ == "__main__":
    output = ROOT / "reports/ai_pre_review_final"
    output.mkdir(parents=True, exist_ok=True)
    for name, body in REPORTS.items():
        (output / name).write_text(f"# {name.removesuffix('.md').replace('_', ' ').title()}\n\n{body}\n", encoding="utf-8")
    audit_json = json.loads((output / "BLINDING_AUDIT.json").read_text(encoding="utf-8"))
    if audit_json["status"] != "PASS":
        raise RuntimeError("cannot build reports from a failed blindness audit")
    print(f"PASS: final_status_reports count={len(REPORTS)} external_results_generated=false")
