#!/usr/bin/env python3
"""Generate dissertation tables directly from Q1 evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "release_evidence/q1_remediation"
CH3 = ROOT / "dissertation_artifacts/q1/chapter3"
CH4 = ROOT / "dissertation_artifacts/q1/chapter4"


def load(path: str) -> dict[str, object]:
    return json.loads((EVIDENCE / path).read_text(encoding="utf-8"))


def csv_table(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                    for key, value in row.items()
                }
            )


def main() -> None:
    h4 = load("uncertainty/h4_uncertainty_hierarchy.json")
    csv_table(CH3 / "table_q1_uncertainty_hierarchy.csv", list(h4["rows"]))
    h1 = load("fidelity/h1_fidelity_noninferiority.json")["summary"]
    csv_table(
        CH4 / "table_q1_fidelity_noninferiority.csv",
        [{"n_pairs": h1["n_pairs"], "mean_delta": h1["mean_difference"], "ci_lower": h1["confidence_interval_95"][0], "ci_upper": h1["confidence_interval_95"][1], "margin": h1["margin"], "status": h1["status"]}],
    )
    h2 = load("traceability/h2_traceability_missingness.json")
    csv_table(
        CH4 / "table_q1_traceability_missingness.csv",
        [{"baseline_k_trace": h2["baseline_k_trace"], "fuzzyxai_k_trace": h2["fuzzyxai_k_trace"], "missingness_f1": h2["missingness"]["f1"], "false_certification": h2["missingness"]["false_certification_rate"]}],
    )
    h3 = load("cascade/h3_adaptive_cascade.json")
    csv_table(CH4 / "table_q1_cascade_policies.csv", list(h3["policies"]))
    h5 = load("critical_rupture/h5_critical_rupture.json")
    csv_table(
        CH4 / "table_q1_critical_rupture.csv",
        [{"structural_f1": h5["structural"]["f1"], "false_certification": h5["structural"]["false_certification_rate"], "incremental_auprc": h5["predictive"]["incremental_auprc"], "predictive_claim_allowed": h5["predictive_claim_allowed"]}],
    )
    h6 = load("rule_ablation/h6_rule_ablation.json")["summary"]
    csv_table(
        CH4 / "table_q1_rule_ablation.csv",
        [{"n_pairs": h6["n_pairs"], "mean_specific_effect": h6["specific_effect"]["mean_difference"], "ci_lower": h6["specific_effect"]["confidence_interval_95"][0], "ci_upper": h6["specific_effect"]["confidence_interval_95"][1], "status": "inconclusive_pending_real_benchmark"}],
    )
    registry = load("datasets/registry.json")
    csv_table(CH4 / "table_q1_dataset_registry.csv", list(registry["real_benchmarks"]))
    print("PASS: q1_tables")


if __name__ == "__main__":
    main()
