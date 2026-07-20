#!/usr/bin/env python3
"""Build release semantics that distinguish null results from false claims."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "release_evidence/q1_final"


def main() -> None:
    claims = json.loads((EVIDENCE / "claim_registry.json").read_text(encoding="utf-8"))
    by_id = {row["claim_id"]: row for row in claims["claims"]}
    external = json.loads((EVIDENCE / "external/status.json").read_text(encoding="utf-8"))
    gates = external["gates"]
    matrix = {
        "schema_version": "2.0",
        "structural_rupture": {
            "status": "PASS" if by_id["H5-structural"]["status"] == "supported" else "FAIL",
            "predictive_status": by_id["H5-predictive"]["status"].upper(),
            "release_blocking": by_id["H5-structural"]["status"] != "supported",
        },
        "rule_ablation": {
            "status": by_id["H6-real-confirmatory"]["status"].upper(),
            "claim_removed_if_not_supported": by_id["H6-real-confirmatory"]["status"] in {"not_supported", "removed"},
            "claim_limited_if_inconclusive": bool(by_id["H6-real-confirmatory"]["limitations"]),
            "release_blocking": (
                by_id["H6-real-confirmatory"]["status"] == "inconclusive"
                and not bool(by_id["H6-real-confirmatory"]["limitations"])
            ),
        },
        "domain_language": {
            "status": gates["domain_language_review"].upper(),
            "release_blocking": gates["domain_language_review"] != "supported",
        },
        "comprehension": {
            "status": gates["comprehension"].upper(),
            "claim_removed_if_not_supported": by_id["H7-comprehension"]["status"] in {"not_supported", "removed"},
            "release_blocking": gates["comprehension"] == "open",
        },
        "expert_action": {
            "status": gates["expert_action_review"].upper(),
            "claim_removed_if_not_supported": by_id["H8-expert-action"]["status"] in {"not_supported", "removed"},
            "release_blocking": gates["expert_action_review"] == "open",
        },
    }
    for key in ("comprehension", "expert_action"):
        section = matrix[key]
        if section["status"] == "NOT_SUPPORTED" and not section["claim_removed_if_not_supported"]:
            section["release_blocking"] = True
    matrix["stable_release_allowed"] = not any(section.get("release_blocking", False) for section in matrix.values() if isinstance(section, dict))
    (EVIDENCE / "final_gate_matrix.json").write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: q1_final_gate_matrix stable={matrix['stable_release_allowed']}")


if __name__ == "__main__":
    main()
