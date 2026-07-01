#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research_validation"
REPORTS = BASE / "reports"
OUTPUTS = BASE / "outputs"
THRESHOLDS = yaml.safe_load((BASE / "configs" / "thresholds.yaml").read_text(encoding="utf-8"))
RESULTS = REPORTS / "research_validation_results.csv"
PACKAGE = REPORTS / "fuzzyxai_research_validation_package.zip"
MANIFEST = REPORTS / "manifest.json"
FIGURES = [
    "rho_by_experiment.png",
    "gamma_delta_scatter.png",
    "action_distribution.png",
    "diagnostic_distribution.png",
    "representation_class_coverage.png",
    "risk_component_heatmap.png",
]
TABLES = [
    "research_validation_results.csv",
    "action_distribution.csv",
    "diagnostic_distribution.csv",
    "representation_class_coverage.csv",
    "risk_component_summary.csv",
]
TRACE_FILES = [
    "route.json",
    "operator_trace.json",
    "operator_table.csv",
    "proof_trace.json",
    "verifier_report.json",
    "dashboard_data.json",
    "operator_dashboard_v2.png",
    "operator_dashboard_v2.html",
    "summary.json",
]


def fail(errors: list[str]) -> int:
    if errors:
        print("research-validation-check: FAIL")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("research-validation-check: PASS")
    return 0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    errors: list[str] = []
    if not RESULTS.exists():
        return fail([f"missing {RESULTS}"])
    rows = list(csv.DictReader(RESULTS.open(encoding="utf-8")))
    total = len(rows)
    task_types = {row["task_type"] for row in rows}
    models = {row["model"] for row in rows}
    actions = {row["action_id"] for row in rows}
    diagnostics = {row["diagnostic_id"] for row in rows}
    representations = {row["representation_class"] for row in rows}
    nonzero = sum(float(row["gamma"]) > 0 and float(row["delta"]) > 0 and float(row["rho"]) > 0 for row in rows)

    checks = [
        (total >= THRESHOLDS["minimum_experiments"], f"expected >= {THRESHOLDS['minimum_experiments']} experiments, got {total}"),
        (len(task_types) >= THRESHOLDS["minimum_task_types"], f"expected >= {THRESHOLDS['minimum_task_types']} task types, got {len(task_types)}"),
        (len(models) >= THRESHOLDS["minimum_models"], f"expected >= {THRESHOLDS['minimum_models']} models, got {len(models)}"),
        (len(actions) >= THRESHOLDS["minimum_actions"], f"expected >= {THRESHOLDS['minimum_actions']} actions, got {len(actions)}"),
        (len(diagnostics) >= THRESHOLDS["minimum_diagnostics"], f"expected >= {THRESHOLDS['minimum_diagnostics']} diagnostics, got {len(diagnostics)}"),
        (len(representations) >= THRESHOLDS["minimum_representation_classes"], f"expected >= {THRESHOLDS['minimum_representation_classes']} representation classes, got {len(representations)}"),
        (all(row["verifier_status"] == "passed" for row in rows), "not all verifier_status values are passed"),
        (all(row["traceability_status"] == "passed" for row in rows), "not all traceability_status values are passed"),
        (all(float(row["rho"]) >= 0 for row in rows), "rho contains negative values"),
        ((nonzero / total) >= THRESHOLDS["minimum_nonzero_ratio"], f"nonzero gamma/delta/rho ratio too low: {nonzero}/{total}"),
    ]
    errors.extend(message for ok, message in checks if not ok)

    for table in TABLES:
        if not (REPORTS / table).exists():
            errors.append(f"missing table {table}")
    for figure in FIGURES:
        path = REPORTS / "figures" / figure
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing figure {figure}")
    if not PACKAGE.exists() or PACKAGE.stat().st_size == 0:
        errors.append("missing fuzzyxai_research_validation_package.zip")
    if not MANIFEST.exists():
        errors.append("missing manifest.json")
    else:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("manifest_self_hash_policy") != "excluded":
            errors.append("manifest_self_hash_policy must be excluded")
        for item in manifest.get("files", []):
            rel = item.get("path", "")
            if rel == "reports/manifest.json":
                errors.append("manifest.json must not be listed in its own sha256 file list")
                continue
            path = BASE / rel
            if not path.exists():
                errors.append(f"manifest file missing: {rel}")
            elif sha256(path) != item.get("sha256"):
                errors.append(f"manifest sha256 mismatch: {rel}")
    for row in rows:
        exp = OUTPUTS / row["experiment_id"]
        for filename in TRACE_FILES:
            path = exp / filename
            if not path.exists() or path.stat().st_size == 0:
                errors.append(f"{row['experiment_id']}: missing {filename}")
        if not (exp / "operator_cards").is_dir():
            errors.append(f"{row['experiment_id']}: missing operator_cards")

    if not errors:
        print(f"experiments={total}; task_types={len(task_types)}; models={len(models)}; actions={len(actions)}; diagnostics={len(diagnostics)}; representations={len(representations)}")
    return fail(errors)


if __name__ == "__main__":
    raise SystemExit(main())
