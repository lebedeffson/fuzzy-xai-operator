#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path.cwd()
    source = root / "artifacts/h10_c3_r4/results/sealed_statistics.json"
    rows = json.loads(source.read_text(encoding="utf-8"))
    by_claim = {row["claim"]: row for row in rows}
    payload = {
        "source": str(source.relative_to(root)),
        "source_values_recalculated": False,
        "primary_endpoint": "H10-C3a",
        "primary_status": "SUPPORTED",
        "primary_effect": by_claim["H10-C3a"]["effect"],
        "key_secondary_endpoint": "cost_regret",
        "key_secondary_status": "SUPPORTED",
        "key_secondary_effect": by_claim["H10-C3a"]["cost_regret_effect"],
        "linked_secondary_endpoint": "H10-C3b",
        "linked_secondary_status": "SUPPORTED",
        "linked_secondary_effect": by_claim["H10-C3b"]["effect"],
        "independent_replication_claim": False,
        "policy_hierarchical_interval": [-0.0051, 0.0073],
        "policy_advantage_supported": False,
        "removed_object_test_value": "historical_archive_only",
        "scope": "pre-generated controlled structural mutations",
    }
    output = root / "reports/chapter_revision/H10_C3_STATISTICAL_HIERARCHY.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output.with_suffix(".md")).write_text(
        "# H10-C3 Statistical Hierarchy\n\n"
        "- Primary endpoint: `H10-C3a`\n"
        "- Key secondary endpoint: `cost_regret`\n"
        "- Linked secondary endpoint: `H10-C3b`\n"
        "- Independent replication claim: `false`\n"
        "- Policy hierarchical 95% CI: `[-0.0051, 0.0073]`\n"
        "- Scope: pre-generated controlled structural mutations\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
