#!/usr/bin/env python3
"""Build leakage-free run-2 input without producing reviewer scores."""

from __future__ import annotations

import hashlib
import json
import zipfile

from common import ROOT, STUDY, sha256, write


SOURCE = ROOT / "study/ai_pre_review_final/public_formative/reviewer_cases.jsonl"
FORBIDDEN = {
    "is_correct",
    "true_label",
    "stratum",
    "expected_action",
    "ground_truth",
    "known_contradictions",
    "known_unsupported_claims",
    "structural_rupture",
}


def forbidden_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(FORBIDDEN & set(value))
        for child in value.values():
            found.update(forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(forbidden_keys(child))
    return found


def main() -> None:
    records = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()]
    cases = {record["case_id"] for record in records}
    if len(cases) != 240 or len(records) != 720:
        raise SystemExit("FAIL: run-2 source must contain 240 cases and three variants per case")
    for record in records:
        leaked = forbidden_keys(record)
        if leaked or record.get("claim_evidence_coverage") != 1.0:
            raise SystemExit(f"FAIL: blind run-2 input invalid: {sorted(leaked)}")
        limitations = record.get("candidate_explanation", {}).get("limitations", [])
        if not limitations:
            raise SystemExit("FAIL: every run-2 card requires an explicit limitation")
    output = STUDY / "ai_formative_run2"
    output.mkdir(parents=True, exist_ok=True)
    input_path = output / "reviewer_cases.jsonl"
    input_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    protocol = {
        "schema_version": "1.0",
        "status": "input_ready_scores_not_run",
        "case_count": len(cases),
        "variant_count": len(records),
        "blind": True,
        "clean_session_required": True,
        "ai_review_is_external_validation": False,
        "acceptance": {
            "critical_unsupported_claims": 0,
            "critical_contradictions": 0,
            "critical_unjustified_actions": 0,
            "minimum_medians": {"uncertainty_honesty": 3, "clarity": 3, "limitation_completeness": 3},
        },
        "reviewer_input_sha256": sha256(input_path),
    }
    write(output / "protocol.json", protocol)
    archive = output / "fuzzyxai-ai-formative-run2-input.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in (input_path, output / "protocol.json"):
            info = zipfile.ZipInfo(path.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, path.read_bytes())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"PASS: final_ai_run2_input cases={len(cases)} variants={len(records)} scores=not_run sha256={digest}")


if __name__ == "__main__":
    main()
