#!/usr/bin/env python3
"""Render concise Q1 Markdown reports from JSON evidence."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "release_evidence/q1_remediation"
REPORTS = ROOT / "reports/q1"


def load(path: str) -> dict[str, object]:
    return json.loads((EVIDENCE / path).read_text(encoding="utf-8"))


def write(name: str, title: str, rows: list[tuple[str, object]], limitations: list[str]) -> None:
    lines = [f"# {title}", "", "Base commit: `cafe403c7d60e36b08f56a5325ba380718a5be35`.", "", "| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {label} | `{value}` |" for label, value in rows)
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in limitations)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    h1 = load("fidelity/h1_fidelity_noninferiority.json")["summary"]
    write(
        "fidelity_noninferiority.md",
        "H1 fidelity non-inferiority",
        [("Pairs", h1["n_pairs"]), ("Mean delta", h1["mean_difference"]), ("95% CI", h1["confidence_interval_95"]), ("Margin", h1["margin"]), ("Status", h1["status"])],
        ["Controlled tabular contour only.", "The system wrapper does not alter the paired local attribution."],
    )
    h2 = load("traceability/h2_traceability_missingness.json")
    write(
        "traceability_and_missingness.md",
        "H2 traceability and missingness",
        [("K_trace baseline", h2["baseline_k_trace"]), ("K_trace FuzzyXAI", h2["fuzzyxai_k_trace"]), ("Missingness F1", h2["missingness"]["f1"]), ("False certification", h2["missingness"]["false_certification_rate"])],
        ["Channels were removed under a controlled protocol."],
    )
    h3 = load("cascade/h3_adaptive_cascade.json")
    write(
        "adaptive_cascade.md",
        "H3 adaptive cascade",
        [("Cost fraction of full", h3["adaptive_cost_fraction_of_full"]), ("Risk margin", h3["risk_noninferiority_margin"]), ("Status", h3["status"])],
        ["Decision costs were preregistered; external operational costs are not claimed."],
    )
    h6 = load("rule_ablation/h6_rule_ablation.json")["summary"]
    write(
        "rule_ablation_conditional.md",
        "H6 conditional rule ablation",
        [("Pairs", h6["n_pairs"]), ("Specific effect", h6["specific_effect"]["mean_difference"]), ("95% CI", h6["specific_effect"]["confidence_interval_95"]), ("Interpretation", "inconclusive pending independent benchmark")],
        ["Controlled leaf-rule candidate; not a general rule-removal effect."],
    )
    h5 = load("critical_rupture/h5_critical_rupture.json")
    write(
        "critical_rupture_structural.md",
        "H5 critical rupture",
        [("Structural F1", h5["structural"]["f1"]), ("False certification", h5["structural"]["false_certification_rate"]), ("Incremental AUPRC", h5["predictive"]["incremental_auprc"]), ("Allowed interpretation", h5["allowed_interpretation"])],
        ["Structural and predictive functions are reported separately."],
    )
    h4 = load("uncertainty/h4_uncertainty_hierarchy.json")
    write(
        "uncertainty_hierarchy.md",
        "H4 uncertainty hierarchy",
        [("Adaptive FML fraction", h4["adaptive_fml_fraction"]), ("Complexity reduction", h4["complexity_reduction_vs_fml"]), ("Non-inferior", h4["non_inferior_to_fml"])],
        ["Uncertainty profiles are controlled injections."],
    )
    sensitivity = load("sensitivity/sensitivity.json")
    write(
        "sensitivity.md",
        "Sensitivity",
        [("K_rob", sensitivity["K_rob"]), ("Parameter points", len(sensitivity["parameter_points"])), ("Threshold points", len(sensitivity["threshold_sweep"]))],
        ["Controlled contour; unstable objects are retained in evidence."],
    )
    scaling = load("scalability/scalability.json")
    write(
        "scalability.md",
        "Scalability",
        [("Sizes", [row["n_objects"] for row in scaling["measurements"]]), ("Log-log slope", scaling["log_log_fit"]["slope"]), ("R squared", scaling["log_log_fit"]["r_squared"]), ("Claim", scaling["complexity_wording"])],
        ["The claim is restricted to measured evidence-graph construction and serialization."],
    )
    write(
        "external_studies.md",
        "External studies",
        [("Comprehension", "planned_not_run"), ("Expert action review", "planned_not_run"), ("Domain language", "pending_external_review")],
        ["No human result is synthesized by code."],
    )
    registry = load("claim_registry.json")
    write(
        "final_q1_validation.md",
        "Q1 remediation status",
        [("Supported controlled claims", registry["supported"]), ("Not supported", registry["not_supported"]), ("Inconclusive", registry["inconclusive"]), ("External gates", registry["external_gate"]), ("Stable release", "BLOCKED")],
        ["Real multimodal benchmark and heavy CI evidence remain separate gates.", "Human gates remain open."],
    )
    print("PASS: q1_reports")


if __name__ == "__main__":
    main()
