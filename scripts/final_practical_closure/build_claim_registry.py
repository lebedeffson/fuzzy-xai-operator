#!/usr/bin/env python3
"""Build a claim-scoped registry that preserves negative predecessor evidence."""

from __future__ import annotations

from common import CONFIRMATORY, FORMATIVE, IMMUTABLE_RESULTS, STUDY, load_json, sha256, write_json


NEW_CLAIMS = ("H3-P1", "H3-P2", "H3-P3", "H3-P4", "H5-A", "H6-A", "H6-B", "H7-A", "H7-B", "H8", "H9")


def main() -> None:
    external = load_json(STUDY / "external_claim_gates.json")
    confirmatory_path = CONFIRMATORY / "summary.json"
    measured: dict[str, dict[str, object]] = {}
    complete = confirmatory_path.is_file()
    if complete:
        summary = load_json(confirmatory_path)
        complete = summary.get("confirmatory_run_completed") is True
        measured = {str(item["experiment_id"]): item for item in summary.get("experiments", []) if isinstance(item, dict)}
    formative = load_json(FORMATIVE / "summary.json")
    claims = []
    for claim_id in NEW_CLAIMS:
        record = measured.get(claim_id)
        claims.append(
            {
                "claim_id": claim_id,
                "enabled": True,
                "status": record.get("status") if record else "not_run",
                "phase": "confirmatory" if record else "formative_only",
                "evidence": None
                if not record
                else {"path": record["artifact_path"], "sha256": record["sha256"]},
                "positive_wording_allowed": bool(record and record.get("status") == "supported"),
            }
        )
    registry = {
        "schema_version": "3.0",
        "confirmatory_run_completed": complete,
        "immutable_original_results": IMMUTABLE_RESULTS,
        "new_claims": claims,
        "external_claims": {
            "enabled": external["claims_enabled"],
            "removed_out_of_scope": external["claims_removed"],
            "gates": {
                "domain_language": external["domain_language"],
                "comprehension": external["comprehension"],
                "expert_action": external["expert_action"],
            },
            "technical_release_blocked": bool(external["claims_enabled"]),
        },
        "formative_summary_sha256": sha256(FORMATIVE / "summary.json"),
        "confirmatory_summary_sha256": sha256(confirmatory_path) if confirmatory_path.is_file() else None,
        "technical_release_allowed": bool(complete and not external["claims_enabled"]),
        "stable_claim_scope": "computational evidence only; no human comprehension, expert, domain-safety or specialist-superiority claim",
        "formative_profile": formative["profile"],
    }
    write_json(FORMATIVE.parent / "claim_registry.json", registry)
    print(f"PASS: practical_claim_registry claims={len(claims)} confirmatory_complete={str(complete).lower()} technical_release={str(registry['technical_release_allowed']).lower()}")


if __name__ == "__main__":
    main()

