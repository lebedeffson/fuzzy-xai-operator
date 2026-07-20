#!/usr/bin/env python3
"""Reject premature human, domain, safety or stable-release claims."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    path = ROOT / "release_evidence/ai_pre_review_final/claim_registry_3.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["stable_release_allowed"]:
        raise RuntimeError("stable release cannot be enabled while external claims are open")
    external = {row["claim_id"]: row["status"] for row in payload["claims"] if row["claim_id"].startswith(("H10", "H11", "H12", "H13"))}
    if not external or any(status != "open_external" for status in external.values()):
        raise RuntimeError(f"premature external claim status: {external}")
    if any(row["status"] == "supported" and not row["allowed_wording"] for row in payload["claims"]):
        raise RuntimeError("supported claim lacks allowed wording")
    print(f"PASS: final_claim_guard open_external={len(external)} stable=false")
