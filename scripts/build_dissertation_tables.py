#!/usr/bin/env python3
"""Build Chapter 3/4 tables strictly from empirical JSON evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "release_evidence/full_empirical_validation"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: Sequence[Mapping[str, object]], title: str) -> None:
    fields = list(rows[0])
    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(evidence: Path, output_root: Path) -> dict[str, object]:
    chapter3 = output_root / "chapter3"
    chapter4 = output_root / "chapter4"
    chapter3.mkdir(parents=True, exist_ok=True)
    chapter4.mkdir(parents=True, exist_ok=True)
    hierarchy = read(evidence / "uncertainty_hierarchy/hierarchy_results.json")
    hierarchy_rows = [
        {
            "mode": row["mode"],
            "coverage": f"{float(row['coverage']):.6f}",
            "undercoverage": f"{float(row['undercoverage']):.6f}",
            "mean_complexity": f"{float(row['mean_complexity']):.6f}",
            "mean_risk": f"{float(row['mean_risk']):.6f}",
        }
        for row in hierarchy["rows"]
    ]
    write_csv(chapter3 / "table_3_uncertainty_hierarchy.csv", hierarchy_rows)
    write_markdown(chapter3 / "table_3_uncertainty_hierarchy.md", hierarchy_rows, "Uncertainty hierarchy comparison")

    multimodal = read(evidence / "empirical_validation/multimodal_results.json")
    model_rows = []
    for run in multimodal["runs"]:
        model_rows.append(
            {
                "dataset": run["dataset_id"],
                "modality": run["modality"],
                "model": run["model_id"],
                "accuracy": f"{float(run['metrics']['accuracy']):.6f}",
                "balanced_accuracy": f"{float(run['metrics']['balanced_accuracy']):.6f}",
                "fit_seconds": f"{float(run['fit_seconds']):.6f}",
                "status": run["status"],
            }
        )
    write_csv(chapter4 / "table_4_multimodal_models.csv", model_rows)
    write_markdown(chapter4 / "table_4_multimodal_models.md", model_rows, "Measured multimodal model runs")

    policies = read(evidence / "policies/policy_comparison.json")
    policy_rows = [
        {
            "cost_scenario": row["cost_scenario"],
            "policy": row["policy_id"],
            "automatic_coverage": f"{float(row['automatic_coverage']):.6f}",
            "critical_wrong_automatic": row["critical_wrong_automatic"],
            "false_blocks": row["false_blocks"],
            "mean_cost": f"{float(row['mean_cost']):.6f}",
        }
        for row in policies["policies"]
    ]
    write_csv(chapter4 / "table_4_policy_comparison.csv", policy_rows)
    write_markdown(chapter4 / "table_4_policy_comparison.md", policy_rows, "Decision-policy comparison")

    baselines_path = evidence / "baselines/baseline_quality_matrix.csv"
    (chapter4 / "table_4_explanation_baselines.csv").write_bytes(baselines_path.read_bytes())
    ablation_path = evidence / "rule_ablation/repeated_cv_metrics.csv"
    (chapter4 / "table_4_rule_ablation_repeated_cv.csv").write_bytes(ablation_path.read_bytes())

    generated = [*chapter3.glob("table_*"), *chapter4.glob("table_*")]
    manifest = {
        "schema_version": "1.0",
        "commit": os.environ.get("FUZZYXAI_COMMIT") or _git_commit(),
        "source_evidence": str(evidence),
        "tables": {
            str(path.relative_to(output_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(generated)
        },
    }
    (output_root / "tables_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: dissertation_tables count={len(generated)}")
    return manifest


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=ROOT / "dissertation_artifacts")
    args = parser.parse_args()
    build(args.evidence, args.output)
