#!/usr/bin/env python3
"""Build claim-safe Q1 registry from measured protocol outputs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "release_evidence/q1_remediation"


def read(path: str) -> dict[str, object]:
    return json.loads((EVIDENCE / path).read_text(encoding="utf-8"))


def main() -> None:
    h1 = read("fidelity/h1_fidelity_noninferiority.json")["summary"]
    h2 = read("traceability/h2_traceability_missingness.json")
    h3 = read("cascade/h3_adaptive_cascade.json")
    h4 = read("uncertainty/h4_uncertainty_hierarchy.json")
    h5 = read("critical_rupture/h5_critical_rupture.json")
    h6 = read("rule_ablation/h6_rule_ablation.json")["summary"]
    rows = [
        claim(
            "H1-01",
            "The FuzzyXAI system layer preserves paired local fidelity within the preregistered margin on the controlled contour.",
            "supported" if h1["noninferior"] else "not_supported",
            {"mean_delta": h1["mean_difference"], "lower_ci_95": h1["lower_bound"], "margin": h1["margin"]},
            ["fidelity/h1_fidelity_noninferiority.json"],
            ["Controlled tabular contour; external modalities require heavy benchmark evidence."],
            str(h1["allowed_wording"]),
            "FuzzyXAI improves the causal correctness of SHAP or any other attribution method.",
        ),
        claim(
            "H2-01",
            "Typed provenance increases traceability and detects controlled missing channels.",
            str(h2["status"]),
            {
                "traceability_gain": h2["traceability_gain"],
                "missingness_f1": h2["missingness"]["f1"],
                "false_certification_rate": h2["missingness"]["false_certification_rate"],
            },
            ["traceability/h2_traceability_missingness.json"],
            ["Missing channels were controlled injections."],
            "On the controlled removal protocol, FuzzyXAI preserved complete provenance and localized missing channels.",
            "FuzzyXAI detects every real-world documentation defect.",
        ),
        claim(
            "H3-01",
            "The adaptive cascade reduces controlled analysis cost at non-inferior decision risk.",
            str(h3["status"]),
            {"cost_fraction": h3["adaptive_cost_fraction_of_full"], "risk_margin": h3["risk_noninferiority_margin"]},
            ["cascade/h3_adaptive_cascade.json"],
            ["Policy signals are controlled; external risk calibration is not established."],
            "The adaptive cascade met the preregistered controlled risk-cost criterion.",
            "The cascade is universally safer or cheaper in production.",
        ),
        claim(
            "H4-01",
            "Adaptive uncertainty representation is non-inferior to always-FML and less complex on injected profiles.",
            "supported" if h4["claim_allowed"] else "not_supported",
            {
                "fml_fraction": h4["adaptive_fml_fraction"],
                "complexity_reduction": h4["complexity_reduction_vs_fml"],
                "epsilon": h4["non_inferiority_epsilon"],
            },
            ["uncertainty/h4_uncertainty_hierarchy.json"],
            ["Profiles have controlled known uncertainty types."],
            str(h4["allowed_wording"]),
            str(h4["forbidden_wording"]),
        ),
        claim(
            "H5-S-01",
            "Critical rupture localizes controlled structural route violations.",
            str(h5["structural_status"]),
            dict(h5["structural"]),
            ["critical_rupture/h5_critical_rupture.json"],
            ["Structural defects were controlled injections."],
            "Critical rupture is a structural diagnostic indicator on the controlled route-violation protocol.",
            "Critical rupture is a predictive safety signal.",
        ),
        claim(
            "H5-P-01",
            "Critical rupture adds held-out predictive value for model error.",
            "supported" if h5["predictive_claim_allowed"] else "not_supported",
            dict(h5["predictive"]),
            ["critical_rupture/h5_critical_rupture.json"],
            ["Predictive addition is secondary and must be positive on held-out data."],
            str(h5["allowed_interpretation"]),
            "Critical rupture improves safety or predicts errors when incremental AUPRC is non-positive.",
        ),
        claim(
            "H6-01",
            "Subgroup-specific leaf-rule ablation exceeds a matched random leaf ablation.",
            "inconclusive",
            {
                "n_pairs": h6["n_pairs"],
                "mean_specific_effect": h6["specific_effect"]["mean_difference"],
                "ci_95": h6["specific_effect"]["confidence_interval_95"],
            },
            ["rule_ablation/h6_rule_ablation.json"],
            ["Positive controlled candidate requires confirmation on a separate real benchmark."],
            "The controlled contour identified a context-dependent candidate effect for validation on independent data.",
            "Rule removal has a general beneficial or harmful effect.",
        ),
        claim(
            "H7-01",
            "FuzzyXAI improves user comprehension without increasing overtrust.",
            "external_gate",
            {"participants": 0},
            ["external_studies/status.json"],
            ["No participant study has been run."],
            "Human usefulness remains unverified pending an independent study.",
            "Users understand or trust FuzzyXAI better.",
        ),
    ]
    payload = {
        "schema_version": "1.0",
        "base_commit": "cafe403c7d60e36b08f56a5325ba380718a5be35",
        "claims": rows,
        "supported": sum(row["status"] == "supported" for row in rows),
        "not_supported": sum(row["status"] == "not_supported" for row in rows),
        "inconclusive": sum(row["status"] == "inconclusive" for row in rows),
        "external_gate": sum(row["status"] == "external_gate" for row in rows),
        "stable_release_allowed": False,
    }
    output = EVIDENCE / "claim_registry.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Q1 claim registry", "", "| Claim | Status | Allowed wording |", "|---|---|---|"]
    lines.extend(f"| {row['claim_id']} | {row['status']} | {row['allowed_wording']} |" for row in rows)
    (EVIDENCE / "claim_registry.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PASS: q1_claim_registry claims={len(rows)}")


def claim(
    claim_id: str,
    text: str,
    status: str,
    metrics: dict[str, object],
    evidence: list[str],
    limitations: list[str],
    allowed: str,
    forbidden: str,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "claim": text,
        "status": status,
        "datasets": ["controlled_tabular_risk_v1"],
        "methods": ["FuzzyXAI"],
        "metrics": metrics,
        "confidence_intervals": {key: value for key, value in metrics.items() if "ci" in key},
        "evidence": evidence,
        "limitations": limitations,
        "allowed_wording": allowed,
        "forbidden_wording": forbidden,
    }


if __name__ == "__main__":
    main()
