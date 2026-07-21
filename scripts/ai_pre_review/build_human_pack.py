#!/usr/bin/env python3
"""Commit AI scores and build packets that conceal every AI result."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from fuzzyxai.ai_pre_review import AI_RUN_IDS, aggregate_ai_reviews, sha256_file
from fuzzyxai.ai_pre_review.contracts import StudyBoundaryError, canonical_json, sha256_bytes

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    lock_path = ROOT / "study/ai_pre_review/confirmatory_protocol_lock.json"
    if not lock_path.is_file():
        raise StudyBoundaryError("human pack requires a frozen confirmatory protocol")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "locked":
        raise StudyBoundaryError("confirmatory protocol is not locked")
    raw_base = ROOT / "release_evidence/ai_pre_review/raw_ai/confirmatory"
    run_paths = {run: raw_base / f"{run}.jsonl" for run in AI_RUN_IDS}
    aggregate = aggregate_ai_reviews(ROOT, run_paths)
    output = ROOT / "study/ai_pre_review/human_confirmation"
    commitment = output / "ai_scores_commitment.json"
    if commitment.exists():
        raise StudyBoundaryError("AI score commitment already exists and cannot be overwritten")
    payload = {
        "schema_version": "1.0",
        "status": "committed_before_human_distribution",
        "protocol_sha256": lock["protocol_sha256"],
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "raw_ai_reviews": {run: sha256_file(path) for run, path in run_paths.items()},
        "aggregate_sha256": sha256_bytes(canonical_json(aggregate).encode()),
        "committed_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["commitment_sha256"] = sha256_bytes(canonical_json(payload).encode())
    output.mkdir(parents=True, exist_ok=True)
    commitment.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    packets = output / "expert_packets"
    if packets.exists():
        raise StudyBoundaryError("expert packets already exist")
    source = ROOT / "study/ai_pre_review/chatgpt_batches/confirmatory"
    shutil.copytree(source, packets)
    assignment = {
        "schema_version": "1.0",
        "status": "ready_for_external_assignment",
        "min_experts": 3,
        "cases_per_expert": 120,
        "variants_per_expert": 360,
        "ai_scores_visible": False,
        "method_identity_visible": False,
        "packets_sha256": {path.name: sha256_file(path) for path in sorted(packets.glob("*.jsonl"))},
    }
    (output / "assignment_manifest.json").write_text(json.dumps(assignment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS: human_packets_ready experts>=3 cases_per_expert=120 ai_scores_hidden=true")


if __name__ == "__main__":
    main()
