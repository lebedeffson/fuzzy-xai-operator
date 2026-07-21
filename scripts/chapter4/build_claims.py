#!/usr/bin/env python3
"""Build claim-safe Chapter 4 statuses from measured formative evidence."""

from __future__ import annotations

import json

from common import EXPERIMENTS, OUTPUT, evidence_ref, load_experiment, prepare


ORIGINAL = {
    "H3-original": "not_supported",
    "H5-P-original": "not_supported",
    "H6-general": "not_supported",
}


def main() -> None:
    prepare()
    claims = []
    for experiment in EXPERIMENTS:
        payload = load_experiment(experiment)
        claims.append(
            {
                "claim_id": experiment.replace("_", "-"),
                "phase": "formative",
                "formative_target_met": bool(payload.get("formative_target_met", False)),
                "confirmatory_status": "not_run",
                "claim_allowed": False,
                "allowed_wording": "Formative measurement only; independent confirmation has not been run.",
                "forbidden_wording": "The hypothesis is confirmed or the method is superior.",
                "evidence": evidence_ref(experiment),
            }
        )
    output = {
        "schema_version": "1.0",
        "chapter_status": "formative_shell_only",
        "original_negative_results": ORIGINAL,
        "new_claims": claims,
        "final_chapter_allowed": False,
    }
    path = OUTPUT / "chapter4_claims.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: chapter4_formative_claims claims={len(claims)} final_allowed=false")


if __name__ == "__main__":
    main()
