#!/usr/bin/env python3
"""Lock the confirmatory protocol only after an accepted formative cycle."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from fuzzyxai.ai_pre_review import sha256_file
from fuzzyxai.ai_pre_review.contracts import StudyBoundaryError, canonical_json, sha256_bytes


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    evidence = ROOT / "release_evidence/ai_pre_review/formative_acceptance.json"
    if not evidence.is_file():
        raise StudyBoundaryError("formative acceptance is not available; confirmatory lock remains open")
    formative = json.loads(evidence.read_text(encoding="utf-8"))
    required = {
        "critical_unsupported_claims": 0,
        "critical_contradictions": 0,
        "critical_unjustified_actions": 0,
        "critical_causal_overclaims": 0,
    }
    if any(formative.get(key) != value for key, value in required.items()):
        raise StudyBoundaryError("formative critical-defect gate did not pass")
    if min(float(formative.get("median_overall_usability", 0)), float(formative.get("median_uncertainty_honesty", 0)), float(formative.get("median_factual_consistency", 0))) < 3:
        raise StudyBoundaryError("formative median score gate did not pass")
    output = ROOT / "study/ai_pre_review/confirmatory_protocol_lock.json"
    if output.exists():
        raise StudyBoundaryError("confirmatory protocol is already locked")
    manifest = ROOT / "study/ai_pre_review/batch_manifest.json"
    confirmatory = b"".join(
        line.encode() + b"\n"
        for line in (ROOT / "study/ai_pre_review/master_explanation_log.jsonl").read_text(encoding="utf-8").splitlines()
        if '"split":"confirmatory"' in line
    )
    payload = {
        "protocol_version": "1.0",
        "status": "locked",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "rubric_sha256": sha256_file(ROOT / "study/ai_pre_review/rubric_v1.yaml"),
        "template_sha256": sha256_file(ROOT / "study/ai_pre_review/ai_review_schema.json"),
        "dictionary_sha256": sha256_bytes(b"no_domain_dictionary_external_domain_knowledge_forbidden"),
        "confirmatory_cases_sha256": sha256_bytes(confirmatory),
        "batch_manifest_sha256": sha256_file(manifest),
        "primary_metrics": ["weighted_kappa", "spearman_overall_usability", "critical_precision", "critical_recall", "preferred_variant_agreement"],
        "critical_flags": json.loads((ROOT / "study/ai_pre_review/claim_registry.json").read_text(encoding="utf-8"))["forbidden_before_human_confirmation"],
        "ai_run_count": 3,
        "human_expert_plan": {"min_experts": 3, "cases_per_expert": 120},
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "locked_by": "repository_protocol_gate",
    }
    payload["protocol_sha256"] = sha256_bytes(canonical_json(payload).encode())
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: confirmatory_protocol_locked sha256={payload['protocol_sha256']}")


if __name__ == "__main__":
    main()
