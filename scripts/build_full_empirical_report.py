#!/usr/bin/env python3
"""Build concise claim-safe Markdown reports for E1-E8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "release_evidence/full_empirical_validation"
DEFAULT_REPORTS = ROOT / "reports/empirical_validation"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build(evidence: Path, reports: Path) -> list[Path]:
    reports.mkdir(parents=True, exist_ok=True)
    manifest = read(evidence / "run_manifest.json")
    ablation = read(evidence / "rule_ablation/statistical_report.json")
    baselines = read(evidence / "baselines/statistical_comparison.json")
    hierarchy = read(evidence / "uncertainty_hierarchy/hierarchy_results.json")
    rupture = read(evidence / "critical_rupture_scalability/critical_rupture_and_scalability.json")
    outputs: list[Path] = []

    lines = [
        "# Full empirical validation report",
        "",
        "## Git",
        f"- branch: `{manifest['branch']}`",
        f"- commit: `{manifest['commit']}`",
        f"- profile: `{manifest['profile']}`",
        f"- release status: `{manifest['release_status']}`",
        "",
        "## E1-E8",
        "",
        "| Experiment | Technical status | Evidence origin |",
        "|---|---|---|",
    ]
    lines.extend(f"| {row['experiment_id']} | {row['status']} | {row['evidence_status']} |" for row in manifest["experiments"])
    lines.extend(
        [
            "",
            "## Measured conclusions",
            f"- Rule ablation: {ablation['interpretation']}.",
            f"- Required explainer methods measured: `{baselines['all_required_measured']}`.",
            f"- Adaptive FML selection fraction: `{hierarchy['adaptive_fml_fraction']:.6f}`.",
            f"- Hierarchy utility claim allowed: `{hierarchy['practical_hierarchy_claim_allowed']}`.",
            f"- Critical-rupture incremental AUPRC: `{rupture['incremental_auprc_over_best_simple_baseline']:.6f}`.",
            f"- Critical-rupture safety claim allowed: `{rupture['safety_claim_allowed']}`.",
            "",
            "## External gates",
        ]
    )
    lines.extend(f"- `{name}`: `{status}`" for name, status in manifest["external_gates"].items())
    lines.extend(
        [
            "",
            "## Forbidden conclusions",
            "- no clinical safety claim;",
            "- no universal superiority claim;",
            "- no external-domain generalization from controlled datasets;",
            "- no stable release while external gates are incomplete.",
        ]
    )
    path = reports / "full_empirical_validation.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs.append(path)

    focused = {
        "rule_ablation_repeated_cv.md": ["# Repeated rule ablation", "", ablation["interpretation"], "", f"Paired comparisons: `{ablation['n_paired_comparisons']}`."],
        "baseline_comparison.md": ["# Explanation baseline comparison", "", f"Required methods measured: `{baselines['all_required_measured']}`.", "", "Technical adapter passage is not treated as explanation quality."],
        "full_population_analysis.md": ["# Full-population analysis", "", "All objects are evaluated with out-of-fold predictions; selected examples are illustrative only."],
    }
    for filename, content in focused.items():
        target = reports / filename
        target.write_text("\n".join(content) + "\n", encoding="utf-8")
        outputs.append(target)
    print(f"PASS: empirical_reports count={len(outputs)}")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args()
    build(args.evidence, args.reports)
