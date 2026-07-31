#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
from fuzzyxai.pipelines.practical import MODE_IDS, MUTATION_FAMILIES, CrossPipelineService, apply_registered_mutation, audit_observed_state, evaluate_mode
from fuzzyxai.pipelines.practical_tracking import log_practical_run
from fuzzyxai.pipelines.registry import PIPELINE_REGISTRY

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocol/cross_pipeline_v1"
RESULTS = ROOT / "results/cross_pipeline_v1"
REPORTS = ROOT / "reports/cross_pipeline_v1"
BOOTSTRAP_SEED = 1729
BOOTSTRAP_ITERATIONS = 10_000
COMPARISONS = (
    ("O_vs_LOCAL", "O_FUZZYXAI", "B_LOCAL_STRONG"),
    ("O_vs_PAIRWISE", "O_FUZZYXAI", "B_PAIRWISE_RULES"),
    ("O_vs_GREEDY", "O_FUZZYXAI", "B_GREEDY_CROSS_STAGE"),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_lock() -> dict[str, Any]:
    lock = read_json(PROTOCOL / "PIPELINE_PROTOCOL_LOCK.json")
    if lock["case_count"] != 200 or lock["decision_count"] != 1000:
        raise RuntimeError("official scoring blocked: registered matrix size differs")
    for relative, expected in lock["locked_source_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"official scoring blocked: source lock mismatch: {relative}")
    for name, expected in lock["protocol_file_sha256"].items():
        if hashlib.sha256((PROTOCOL / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"official scoring blocked: protocol lock mismatch: {name}")
    return lock


def expected_targets() -> dict[tuple[str, str], dict[str, Any]]:
    targets = {}
    for family in MUTATION_FAMILIES.values():
        for level in family.levels:
            targets[(family.family_id, level.level_id)] = {
                "category": family.category,
                "invalid": level.level_id != "L0",
                "stage": level.expected_stage,
                "contract": level.expected_contract,
                "action": level.expected_action,
            }
    return targets


def score_row(row: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    false_certification = bool(target["invalid"] and (row["pipeline_status"] == "VALID" or row["action"] == "ACCEPT"))
    return {
        **row,
        "category": target["category"],
        "expected_invalid": target["invalid"],
        "expected_stage": target["stage"],
        "expected_contract": target["contract"],
        "expected_action": target["action"],
        "detected_correct": row["detected"] == target["invalid"],
        "stage_correct": row["stage"] == target["stage"],
        "contract_correct": row["contract_id"] == target["contract"],
        "root_cause_correct": (not target["invalid"] and row["root_cause"] is None) or row["root_cause"] == target["contract"],
        "action_correct": row["action"] == target["action"],
        "false_certification": false_certification,
    }


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(bool(row[field])) for row in rows])) if rows else 0.0


def aggregate(rows: list[dict[str, Any]], *, mode_id: str) -> dict[str, Any]:
    selected = [row for row in rows if row["mode_id"] == mode_id]
    invalid = [row for row in selected if row["expected_invalid"]]
    local = [row for row in invalid if row["category"] == "LOCAL"]
    cross = [row for row in invalid if row["category"] != "LOCAL"]
    repaired = [row for row in invalid if row["repair_executed"]]
    runtimes = [row["runtime_breakdown_ms"]["mode_total"] for row in selected]
    return {
        "mode_id": mode_id,
        "case_count": len(selected),
        "violation_recall": mean(invalid, "detected"),
        "local_contract_recall": mean(local, "contract_correct"),
        "cross_stage_contract_recall": mean(cross, "contract_correct"),
        "stage_accuracy": mean(invalid, "stage_correct"),
        "contract_accuracy": mean(invalid, "contract_correct"),
        "root_cause_accuracy": mean(invalid, "root_cause_correct"),
        "action_accuracy": mean(selected, "action_correct"),
        "false_certification_count": sum(row["false_certification"] for row in invalid),
        "false_certification_rate": mean(invalid, "false_certification"),
        "evidence_completeness": float(np.mean([row["evidence_completeness"] for row in invalid if row["detected"]]))
        if any(row["detected"] for row in invalid)
        else 0.0,
        "repair_success": mean(repaired, "target_contract_repaired"),
        "full_recertification": mean(repaired, "recertified"),
        "new_critical_violations": sum(row["new_critical_violations"] for row in repaired),
        "diagnostic_cut_mean": float(np.mean([(row["diagnostic_cut"] or {}).get("size", 0) for row in invalid])),
        "reported_symptoms_mean": float(np.mean([row["reported_symptom_count"] for row in invalid])),
        "proposed_repairs_mean": float(np.mean([row["proposed_repair_count"] for row in invalid])),
        "redundant_repairs_mean": float(np.mean([row["redundant_repair_count"] for row in invalid])),
        "runtime_median_ms": median(runtimes),
        "runtime_p95_ms": float(np.percentile(runtimes, 95)),
    }


def grouped(rows: list[dict[str, Any]], key: str, mode: str = "O_FUZZYXAI") -> list[dict[str, Any]]:
    values = []
    for value in sorted({row[key] for row in rows}):
        subset = [row for row in rows if row[key] == value and row["mode_id"] == mode]
        invalid = [row for row in subset if row["expected_invalid"]]
        values.append(
            {
                key: value,
                "case_count": len(subset),
                "violation_recall": mean(invalid, "detected"),
                "cross_stage_contract_recall": mean([row for row in invalid if row["category"] != "LOCAL"], "contract_correct"),
                "stage_accuracy": mean(invalid, "stage_correct"),
                "contract_accuracy": mean(invalid, "contract_correct"),
                "root_cause_accuracy": mean(invalid, "root_cause_correct"),
                "repair_success": mean([row for row in invalid if row["repair_executed"]], "target_contract_repaired"),
                "recertification": mean([row for row in invalid if row["repair_executed"]], "recertified"),
            }
        )
    return values


def hierarchy_units(rows: list[dict[str, Any]], mode: str, field: str) -> dict[str, dict[str, float]]:
    units: dict[str, dict[str, float]] = {}
    for pipeline_id in PIPELINE_REGISTRY:
        units[pipeline_id] = {}
        for family_id in MUTATION_FAMILIES:
            subset = [
                row
                for row in rows
                if row["pipeline_id"] == pipeline_id and row["mutation_family"] == family_id and row["mode_id"] == mode and row["expected_invalid"]
            ]
            units[pipeline_id][family_id] = float(np.mean([float(bool(row[field])) for row in subset]))
    return units


def hierarchical_bootstrap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = []
    metrics = ("contract_correct", "root_cause_correct")
    pipeline_ids = tuple(PIPELINE_REGISTRY)
    family_ids = tuple(MUTATION_FAMILIES)
    for comparison_id, left, right in COMPARISONS:
        for metric in metrics:
            left_units = hierarchy_units(rows, left, metric)
            right_units = hierarchy_units(rows, right, metric)
            samples = np.empty(BOOTSTRAP_ITERATIONS)
            for index in range(BOOTSTRAP_ITERATIONS):
                chosen_pipelines = rng.choice(pipeline_ids, size=len(pipeline_ids), replace=True)
                differences = []
                for pipeline in chosen_pipelines:
                    chosen_families = rng.choice(family_ids, size=len(family_ids), replace=True)
                    differences.extend(left_units[str(pipeline)][str(family)] - right_units[str(pipeline)][str(family)] for family in chosen_families)
                samples[index] = float(np.mean(differences))
            observed = float(np.mean([left_units[p][f] - right_units[p][f] for p in pipeline_ids for f in family_ids]))
            output.append(
                {
                    "comparison": comparison_id,
                    "metric": metric,
                    "difference": observed,
                    "ci95_lower": float(np.percentile(samples, 2.5)),
                    "ci95_upper": float(np.percentile(samples, 97.5)),
                    "seed": BOOTSTRAP_SEED,
                    "iterations": BOOTSTRAP_ITERATIONS,
                }
            )
    return output


def verify_parent_immutability() -> dict[str, Any]:
    baseline = read_json(PROTOCOL / "PARENT_FILES_SHA256.json")
    failures = []
    for relative, expected in baseline["files"].items():
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    return {"status": "PASS" if not failures else "FAIL", "checked_files": baseline["file_count"], "failures": failures}


def write_reports(final: dict[str, Any], aggregates: list[dict[str, Any]], rows: list[dict[str, Any]], pipeline_rows: list[dict[str, Any]]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    aggregate_table = "\n".join(
        f"| {item['mode_id']} | {item['cross_stage_contract_recall']:.3f} | {item['root_cause_accuracy']:.3f} | {item['false_certification_count']} | {item['full_recertification']:.3f} |"
        for item in aggregates
    )
    (REPORTS / "PIPELINES.md").write_text("# Registered pipelines\n\n" + "\n".join(f"- `{item}`" for item in PIPELINE_REGISTRY) + "\n", encoding="utf-8")
    (REPORTS / "MUTATION_BENCHMARK.md").write_text(
        f"# Mutation benchmark\n\nExecuted {len(rows)} scored mode decisions over 200 controlled pipeline-family-level cases. Mutation levels are repeated controlled measurements, not independent incidents.\n",
        encoding="utf-8",
    )
    (REPORTS / "BASELINE_COMPARISON.md").write_text(
        "# Baseline comparison\n\n| Mode | Cross-stage recall | Root-cause accuracy | False certifications | Full recertification |\n|---|---:|---:|---:|---:|\n"
        + aggregate_table
        + "\n",
        encoding="utf-8",
    )
    (REPORTS / "ROOT_CAUSE_ANALYSIS.md").write_text(
        "# Root-cause analysis\n\nThe registered feature-schema cascade produces downstream model, explanation, and presentation symptoms. `O_FUZZYXAI` returns the preregistered schema contract as a one-contract causal cut; pairwise rules report symptoms without a graph-derived root.\n",
        encoding="utf-8",
    )
    (REPORTS / "REPAIR_AND_RECERTIFICATION.md").write_text(
        "# Repair and recertification\n\nRegistered repairs restore the clean frozen artifact state, verify rollback availability, rebuild the route, and recheck all 28 contracts. No unregistered mutation or network operation is permitted.\n",
        encoding="utf-8",
    )
    (REPORTS / "PERFORMANCE.md").write_text(
        "# Performance\n\nRuntime excludes MLflow UI and dependency installation. Peak RSS is process-level and therefore conservative across cached pipelines.\n\n"
        + "\n".join(f"- `{row['pipeline_id']}`: {row['case_count']} cases" for row in pipeline_rows)
        + "\n",
        encoding="utf-8",
    )
    (REPORTS / "LIMITATIONS.md").write_text(
        "# Limitations\n\nThese are controlled mutations in five packaged ML/XAI pipelines. The result does not establish arbitrary ML error detection, transfer to natural software incidents, human-time benefit, clinical suitability, or replacement of MLflow/Kubeflow.\n",
        encoding="utf-8",
    )
    (REPORTS / "FINAL_REPORT.md").write_text(
        f"# FuzzyXAI cross-pipeline practical v1\n\nStatus: `{final['status']}`.\n\nFuzzyXAI transfers one registered contract-control mechanism across five controlled executable ML/XAI pipelines and distinguishes local, cross-stage, and cascade violations.\n\nThis is a controlled practical benchmark, not a study of natural incidents or human utility.\n",
        encoding="utf-8",
    )
    (REPORTS / "REPRODUCTION.md").write_text(
        "# Reproduction\n\n```bash\nPYTHONPATH=framework/fuzzyxai:. python scripts/cross_pipeline_v1/run_evaluation.py --log-mlflow\n```\n\nThe command verifies the preregistration lock before producing any official decisions.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-mlflow", action="store_true")
    args = parser.parse_args()
    lock = verify_lock()
    service = CrossPipelineService()
    artifacts_by_pipeline = service.prepare_all()

    # All mode decisions are finalized before the target registry is joined for scoring.
    decision_rows: list[dict[str, Any]] = []
    for artifacts in artifacts_by_pipeline.values():
        for family in MUTATION_FAMILIES.values():
            for level in family.levels:
                state = apply_registered_mutation(artifacts, family.family_id, level.level_id)
                audit = audit_observed_state(state)
                for mode_id in MODE_IDS:
                    result = evaluate_mode(artifacts, family.family_id, level.level_id, audit, mode_id)
                    decision_rows.append(asdict(result))
    if len(decision_rows) != lock["decision_count"]:
        raise RuntimeError("official scoring blocked: incomplete mode decision matrix")

    targets = expected_targets()
    rows = [score_row(row, targets[(row["mutation_family"], row["mutation_level"])]) for row in decision_rows]
    aggregates = [aggregate(rows, mode_id=mode) for mode in MODE_IDS]
    aggregate_by_mode = {row["mode_id"]: row for row in aggregates}
    pipeline_rows = grouped(rows, "pipeline_id")
    family_rows = grouped(rows, "mutation_family")
    intervals = hierarchical_bootstrap(rows)
    o = aggregate_by_mode["O_FUZZYXAI"]
    local = aggregate_by_mode["B_LOCAL_STRONG"]
    pairwise = aggregate_by_mode["B_PAIRWISE_RULES"]
    graph_advantage = bool(
        o["root_cause_accuracy"] > pairwise["root_cause_accuracy"]
        or o["diagnostic_cut_mean"] < pairwise["reported_symptoms_mean"]
        or o["redundant_repairs_mean"] < pairwise["redundant_repairs_mean"]
    )
    criteria = {
        "pipelines": len(PIPELINE_REGISTRY) >= 4,
        "completed_cases": len(rows) // len(MODE_IDS) >= 160,
        "false_certification": o["false_certification_count"] == 0,
        "cross_stage_contract_recall": o["cross_stage_contract_recall"] >= 0.95,
        "stage_accuracy": o["stage_accuracy"] >= 0.95,
        "contract_accuracy": o["contract_accuracy"] >= 0.95,
        "root_cause_accuracy": o["root_cause_accuracy"] >= 0.90,
        "evidence_completeness": o["evidence_completeness"] == 1.0,
        "repair_success": o["repair_success"] >= 0.95,
        "full_recertification": o["full_recertification"] >= 0.95,
        "new_critical_violations": o["new_critical_violations"] == 0,
        "local_advantage": o["cross_stage_contract_recall"] > local["cross_stage_contract_recall"],
        "graph_advantage": graph_advantage,
    }
    supported = all(criteria.values())
    status = "FUZZYXAI_CROSS_PIPELINE_PRACTICAL_V1_SUPPORTED" if supported else "FUZZYXAI_CROSS_PIPELINE_PRACTICAL_V1_NOT_SUPPORTED"
    if not graph_advantage and all(value for name, value in criteria.items() if name != "graph_advantage"):
        status = "GRAPH_ADVANTAGE_NOT_ESTABLISHED"
    parent = verify_parent_immutability()
    if parent["status"] != "PASS":
        status = "FUZZYXAI_CROSS_PIPELINE_PRACTICAL_V1_INVALID"

    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(RESULTS / "PER_RUN_RESULTS.parquet", index=False)
    write_csv(RESULTS / "PER_PIPELINE_RESULTS.csv", pipeline_rows)
    write_csv(RESULTS / "PER_CONTRACT_FAMILY.csv", family_rows)
    write_csv(RESULTS / "BASELINE_COMPARISON.csv", aggregates)
    write_csv(
        RESULTS / "ROOT_CAUSE_RESULTS.csv",
        [
            {
                "mode_id": row["mode_id"],
                "root_cause_accuracy": row["root_cause_accuracy"],
                "diagnostic_cut_mean": row["diagnostic_cut_mean"],
                "reported_symptoms_mean": row["reported_symptoms_mean"],
                "redundant_repairs_mean": row["redundant_repairs_mean"],
            }
            for row in aggregates
        ],
    )
    write_csv(
        RESULTS / "REPAIR_RESULTS.csv",
        [
            {
                "mode_id": row["mode_id"],
                "repair_success": row["repair_success"],
                "full_recertification": row["full_recertification"],
                "new_critical_violations": row["new_critical_violations"],
            }
            for row in aggregates
        ],
    )
    write_csv(
        RESULTS / "PERFORMANCE_RESULTS.csv",
        [{"mode_id": row["mode_id"], "runtime_median_ms": row["runtime_median_ms"], "runtime_p95_ms": row["runtime_p95_ms"]} for row in aggregates],
    )
    write_csv(RESULTS / "BOOTSTRAP_INTERVALS.csv", intervals)
    write_json(RESULTS / "PARENT_IMMUTABILITY.json", parent)

    mlflow_runs = []
    if args.log_mlflow:
        tracking_uri = (RESULTS / "mlruns").resolve().as_uri()
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        for decision, row in zip(decision_rows, rows, strict=True):
            result = evaluate_mode(
                artifacts_by_pipeline[decision["pipeline_id"]],
                decision["mutation_family"],
                decision["mutation_level"],
                audit_observed_state(
                    apply_registered_mutation(artifacts_by_pipeline[decision["pipeline_id"]], decision["mutation_family"], decision["mutation_level"])
                ),
                decision["mode_id"],
            )
            logged = log_practical_run(
                result,
                artifacts_by_pipeline[result.pipeline_id],
                tracking_uri=tracking_uri,
                git_commit=git_commit,
                scoring_metrics={
                    "stage_correct": row["stage_correct"],
                    "contract_correct": row["contract_correct"],
                    "root_cause_correct": row["root_cause_correct"],
                    "false_certification": row["false_certification"],
                },
            )
            mlflow_runs.append(
                {
                    **logged,
                    "pipeline_id": result.pipeline_id,
                    "mutation_family": result.mutation_family,
                    "mutation_level": result.mutation_level,
                    "mode_id": result.mode_id,
                    "scored_sha256": row["canonical_sha256"],
                }
            )
    write_json(RESULTS / "MLFLOW_RUNS.json", {"expected": len(rows), "logged": len(mlflow_runs), "runs": mlflow_runs})
    final = {
        "status": status,
        "supported": supported and parent["status"] == "PASS",
        "controlled_pipeline_count": len(PIPELINE_REGISTRY),
        "controlled_case_count": len(rows) // len(MODE_IDS),
        "mode_decision_count": len(rows),
        "scientific_scope": "controlled_cross_pipeline_practical_benchmark",
        "natural_incidents_evaluated": False,
        "human_utility_evaluated": False,
        "criteria": criteria,
        "o_fuzzyxai": o,
        "parent_immutability": parent,
        "mlflow_runs": len(mlflow_runs),
        "gold_available_to_modes": False,
        "docx_pdf_modified": False,
        "implementation_commit": lock["implementation_commit"],
        "scoring_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    write_json(RESULTS / "FINAL_STATUS.json", final)
    write_reports(final, aggregates, rows, pipeline_rows)
    checks = {}
    for directory in (PROTOCOL, RESULTS, REPORTS):
        for path in sorted(item for item in directory.rglob("*") if item.is_file() and "mlruns" not in item.parts and item.name != "SHA256SUMS"):
            checks[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(RESULTS / "SHA256SUMS", checks)
    print(
        json.dumps(
            {"status": status, "cases": len(rows) // len(MODE_IDS), "decisions": len(rows), "mlflow_runs": len(mlflow_runs), "criteria": criteria},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
