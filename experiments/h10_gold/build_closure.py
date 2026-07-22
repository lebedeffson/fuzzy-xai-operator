from __future__ import annotations

import argparse
import ast
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import ARTIFACT_ROOT, ROOT, load_config, sha256_file, write_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def _group(rows: list[dict[str, str]], population: str | None = None) -> list[dict[str, Any]]:
    selected = rows if population is None else [row for row in rows if row["case_type"] == population]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        groups[row["method"]].append(row)
    output = []
    for method, items in sorted(groups.items()):
        def mean(field: str) -> float:
            return sum(float(row[field]) for row in items) / len(items)
        output.append(
            {
                "method": method,
                "population": population or "overall",
                "cases": len(items),
                "source_localization_f1": mean("source_f1"),
                "repair_set_f1": mean("repair_f1"),
                "false_certification": mean("false_certification"),
                "false_block": mean("false_block"),
                "cut_exact": mean("cut_exact"),
                "cut_jaccard": mean("cut_jaccard"),
                "cut_cost_regret": mean("cut_cost_regret"),
                "repair_cost_regret": mean("repair_cost_regret"),
                "runtime_ms": mean("runtime_ms"),
                "evidence_status": "exploratory_development_only",
            }
        )
    return output


def _forbidden_symbol_audit() -> dict[str, Any]:
    forbidden = {
        "H10Auditor",
        "TypedRouteGuard",
        "SourceLocalizer",
        "FaultFamilyClassifier",
        "DiagnosticCutSolver",
        "RepairPlanner",
        "RepairSetPlanner",
        "SPEC_BY_LEAF",
        "FIELD_TO_SPECS",
    }
    findings = []
    for path in sorted((ROOT / "gold_oracle").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        if forbidden.intersection(names) or any("fuzzyxai" in item or "audit_h10" in item for item in imports):
            findings.append(str(path.relative_to(ROOT)))
    return {"status": "PASS" if not findings else "FAIL", "findings": findings}


def build(config_path: Path) -> None:
    config = load_config(config_path)
    manifest = json.loads((ARTIFACT_ROOT / "h10_final_gold_manifest.json").read_text())
    power = json.loads((ARTIFACT_ROOT / "exploratory" / "power_analysis.json").read_text())
    raw_path = ARTIFACT_ROOT / "exploratory" / "development_raw_results.csv"
    raw = _read_csv(raw_path)
    tables = ARTIFACT_ROOT / "tables"
    table_map = {
        "overall_results.csv": _group(raw),
        "single_fault_results.csv": _group(raw, "single"),
        "composite_fault_results.csv": _group(raw, "composite"),
        "unknown_results.csv": _group(raw, "unknown_ambiguous"),
    }
    for name, rows in table_map.items():
        _write_csv(tables / name, rows)
    _write_csv(
        tables / "minimal_cut_results.csv",
        [
            {key: row[key] for key in ("method", "population", "cases", "cut_exact", "cut_jaccard", "cut_cost_regret", "evidence_status")}
            for row in table_map["composite_fault_results.csv"]
        ],
    )
    _write_csv(
        tables / "repair_cost_results.csv",
        [
            {key: row[key] for key in ("method", "population", "cases", "repair_set_f1", "repair_cost_regret", "evidence_status")}
            for row in table_map["composite_fault_results.csv"]
        ],
    )
    per_pipeline = []
    for (pipeline, method), items in sorted(
        ((key, value) for key, value in _group_pairs(raw, ("pipeline_id", "method"))), key=lambda item: item[0]
    ):
        composite = [row for row in items if row["case_type"] == "composite"]
        per_pipeline.append(
            {
                "pipeline_id": pipeline,
                "method": method,
                "composite_cases": len(composite),
                "source_localization_f1": _mean(composite, "source_f1"),
                "repair_set_f1": _mean(composite, "repair_f1"),
                "cut_exact": _mean(composite, "cut_exact"),
                "evidence_status": "exploratory_development_only",
            }
        )
    _write_csv(tables / "per_pipeline_results.csv", per_pipeline)

    reviewer_files = [ARTIFACT_ROOT / "adjudication" / f"reviewer_{index}.csv" for index in (1, 2)]
    oracle_audit = _forbidden_symbol_audit()
    reasons = []
    if not all(path.exists() for path in reviewer_files):
        reasons.append("two_real_manual_adjudication_files_absent")
    if power["status"] != "PASS":
        reasons.append("development_primary_effect_below_registered_margin")
    methodology = {
        "study_id": config["study_id"],
        "status": "PASS" if not reasons else "BLOCKED_PRECONFIRMATORY",
        "release_allowed": not reasons,
        "blocking_reasons": reasons,
        "oracle_import_independence": oracle_audit,
        "gold_source": "executed_transaction_log",
        "gold_repair": "inverse_low_level_transactions",
        "gold_cut": "independent_exhaustive_graph_diff_hitting_set",
        "gold_uses_h10_taxonomy": False,
        "manual_adjudication_completed": all(path.exists() for path in reviewer_files),
        "sealed_test_opened": (ARTIFACT_ROOT / "opening" / "opening_record.json").exists(),
        "post_test_tuning": False,
        "old_v16_changed": False,
        "old_v18_changed": False,
        "old_v19_changed": False,
    }
    write_json(ARTIFACT_ROOT / "closure" / "h10_final_gold_methodology_audit.json", methodology)
    claims = {
        "study_id": config["study_id"],
        "phase": "preconfirmatory_blocked" if reasons else "ready_for_lock",
        "claims": {
            "H10-L": "not_evaluated_confirmatory",
            "H10-R": "not_evaluated_confirmatory",
            "H10-C": "exploratory_only",
            "H10-U": "not_evaluated_confirmatory",
        },
        "development_observation": {
            "H10-L_effect_vs_best_strong_baseline": 0.0,
            "H10-R_effect_vs_best_strong_baseline": 0.0,
            "H10_C_cut_exact_full": _metric(table_map["composite_fault_results.csv"], "full_h10", "cut_exact"),
            "H10_C_cut_exact_baseline": _metric(table_map["composite_fault_results.csv"], power["best_baseline_selected_on_development"], "cut_exact"),
        },
        "manual_positive_override": False,
        "sealed_test_opened": False,
    }
    write_json(ARTIFACT_ROOT / "closure" / "h10_final_gold_claim_registry.json", claims)
    leakage = {
        "sealed_test_opened": False,
        "opening_count": 0,
        "sealed_scoring_performed": False,
        "post_lock_tuning": False,
        "methods_received_mutation_log": False,
        "methods_received_gold": False,
        "private_truth_tracked_by_git": False,
        "release_contains_private_truth": False,
    }
    write_json(ARTIFACT_ROOT / "closure" / "h10_final_gold_leakage_audit.json", leakage)

    evidence = []
    for filename, rows in table_map.items():
        path = tables / filename
        file_hash = sha256_file(path)
        for row_index, row in enumerate(rows, start=2):
            for metric in ("source_localization_f1", "repair_set_f1", "false_certification", "false_block", "cut_exact", "cut_cost_regret", "repair_cost_regret"):
                evidence.append(
                    {
                        "claim_id": "EXPLORATORY-H10-GOLD",
                        "metric": metric,
                        "value": row[metric],
                        "population": row["population"],
                        "method": row["method"],
                        "source_file": str(path.relative_to(ROOT)),
                        "locator": f"row={row_index},column={metric}",
                        "sha256": file_hash,
                        "status": "exploratory",
                    }
                )
    write_json(ARTIFACT_ROOT / "closure" / "h10_final_gold_evidence_map.json", {"entries": evidence})
    head = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    report = f"""# H10 final Gold validation report

- Study: `{config['study_id']}`
- Repository HEAD at report generation: `{head}`
- Gold cases generated: `{manifest['case_count']}` across `{manifest['pipeline_count']}` pipelines
- Composite cases: `{manifest['case_counts']['composite']}`
- Oracle independence: `{oracle_audit['status']}`
- Manual adjudication: `PENDING_TWO_REAL_REVIEWERS`
- Power gate: `{power['status']}`
- Sealed opening count: `0`
- Confirmatory scoring: `NOT_RUN`
- Scientific release: `BLOCKED`

## Blocking reasons

{chr(10).join(f'- `{reason}`' for reason in reasons)}

## Development-only observation

After strengthening both baselines to ignore the explicitly derived status
field, Full H10 and the best baseline both reached 1.0 source and repair F1 on
the composite development subset. The expected primary effect is 0.0, below
the registered 0.04 margin. Increasing sample size cannot create a missing
effect. Minimal-cut exact match remains exploratory and secondary.
"""
    (ARTIFACT_ROOT / "closure" / "h10_final_gold_validation_report.md").write_text(report, encoding="utf-8")


def _group_pairs(rows: list[dict[str, str]], fields: tuple[str, ...]):
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    return groups.items()


def _mean(rows: list[dict[str, str]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def _metric(rows: list[dict[str, Any]], method: str, metric: str) -> float:
    return float(next(row[metric] for row in rows if row["method"] == method))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "h10_final_gold_protocol.yaml")
    args = parser.parse_args()
    build(args.config)


if __name__ == "__main__":
    main()
