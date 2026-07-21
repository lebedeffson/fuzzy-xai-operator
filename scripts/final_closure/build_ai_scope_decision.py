#!/usr/bin/env python3
"""Record the non-human AI text-review boundary for the technical release."""

from __future__ import annotations

from common import ROOT, STUDY, sha256, write


FORBIDDEN_HUMAN_CLAIMS = (
    "understandable_to_users",
    "confirmed_by_experts",
    "improves_domain_safety",
    "matches_specialist_decisions",
)


def main() -> None:
    review_bundle = STUDY / "ai_formative_run2/fuzzyxai-ai-formative-run2-input.zip"
    reviewer_cases = STUDY / "ai_formative_run2/reviewer_cases.jsonl"
    if not review_bundle.is_file() or not reviewer_cases.is_file():
        raise SystemExit("FAIL: AI text-review input is incomplete")
    case_count = sum(1 for _ in reviewer_cases.open(encoding="utf-8"))
    if case_count != 720:
        raise SystemExit(f"FAIL: expected 720 blind variants, found {case_count}")
    payload = {
        "schema_version": "1.0",
        "status": "not_run_not_blocking_technical_release",
        "scope": "automated_formative_text_review_only",
        "review_completed": False,
        "review_records": 0,
        "ai_review_is_external_validation": False,
        "technical_release_may_proceed": True,
        "condition": "all human-perception and expert-validation claims are disabled",
        "disabled_claims": list(FORBIDDEN_HUMAN_CLAIMS),
        "external_gates": {
            "domain_language": "open_external_not_in_scope",
            "comprehension": "open_external_not_in_scope",
            "expert_action": "open_external_not_in_scope",
        },
        "input_bundle": {
            "path": review_bundle.relative_to(ROOT).as_posix(),
            "sha256": sha256(review_bundle),
            "blind_variants": case_count,
        },
        "limitations": [
            "No AI text-quality acceptance result is claimed.",
            "No human comprehension, domain correctness or expert-action claim is enabled.",
            "A later AI review may improve wording but cannot change frozen models, features or endpoints.",
        ],
    }
    write(STUDY / "ai_text_review_scope.json", payload)
    print("PASS: final_ai_text_review_scope review=not_run human_claims=disabled")


if __name__ == "__main__":
    main()
