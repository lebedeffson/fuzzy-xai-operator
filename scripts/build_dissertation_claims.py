#!/usr/bin/env python3
"""Derive claim-safe Chapter 3/4 statements from measured evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "release_evidence/full_empirical_validation"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build(evidence: Path, output: Path) -> dict[str, object]:
    hierarchy = read(evidence / "uncertainty_hierarchy/hierarchy_results.json")
    rupture = read(evidence / "critical_rupture_scalability/critical_rupture_and_scalability.json")
    ablation = read(evidence / "rule_ablation/statistical_report.json")
    baselines = read(evidence / "baselines/statistical_comparison.json")
    claims = [
        {
            "claim_id": "ch3.adaptive_hierarchy_practical_utility",
            "claim": "Adaptive uncertainty selection reduces representation complexity without unacceptable measured risk.",
            "status": "supported" if hierarchy["practical_hierarchy_claim_allowed"] and hierarchy["non_inferior_to_fml"] else "not_supported",
            "evidence_files": ["uncertainty_hierarchy/hierarchy_results.json"],
            "metrics": {"adaptive_fml_fraction": hierarchy["adaptive_fml_fraction"], "non_inferior": hierarchy["non_inferior_to_fml"]},
            "limitations": ["controlled uncertainty injection"],
        },
        {
            "claim_id": "ch4.rule_ablation_general_effect",
            "claim": ablation["interpretation"],
            "status": "supported" if ablation["interpretation"].startswith("rule removal systematically") else "inconclusive",
            "evidence_files": ["rule_ablation/statistical_report.json", "rule_ablation/repeated_cv_metrics.csv"],
            "metrics": ablation["statistics"]["subgroup_recall"],
            "limitations": ablation["limitations"],
        },
        {
            "claim_id": "ch4.critical_rupture_predictive_value",
            "claim": "Critical rupture provides incremental predictive information about wrong automatic decisions.",
            "status": "supported" if rupture["safety_claim_allowed"] else "not_supported",
            "evidence_files": ["critical_rupture_scalability/critical_rupture_and_scalability.json"],
            "metrics": {"incremental_auprc": rupture["incremental_auprc_over_best_simple_baseline"]},
            "limitations": [rupture["claim_rule"]],
        },
        {
            "claim_id": "ch4.explainer_comparison",
            "claim": "Required external explainers were measured under the common protocol.",
            "status": "supported" if baselines["all_required_measured"] else "not_supported",
            "evidence_files": ["baselines/baseline_quality_matrix.csv", "baselines/statistical_comparison.json"],
            "metrics": {"n_explained": baselines["n_explained"]},
            "limitations": baselines["limitations"],
        },
        {
            "claim_id": "ch4.external_comprehension",
            "claim": "Independent users understand the generated explanations.",
            "status": "not_supported",
            "evidence_files": ["external_review/review_status.json"],
            "metrics": {},
            "limitations": ["comprehension pilot and expert review are planned_not_run"],
        },
    ]
    payload = {"schema_version": "1.0", "claims": claims, "forbidden_claims": ["clinical safety", "universal superiority", "external validity from controlled generators"]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "chapter3_4_claims.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Measured Chapter 3/4 claims", ""]
    lines.extend(f"- `{item['status']}`: {item['claim']}" for item in claims)
    (output / "chapter3_4_claims.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PASS: dissertation_claims count={len(claims)}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=ROOT / "dissertation_artifacts/claims")
    args = parser.parse_args()
    build(args.evidence, args.output)
