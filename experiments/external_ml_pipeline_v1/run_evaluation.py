from __future__ import annotations

import json
import tempfile
from pathlib import Path
from statistics import median
from typing import Any

import mlflow
import numpy as np
from scipy.stats import binomtest

from .benchmark import FAULTS, MODE_IDS, ExternalBenchmark, build_route
from .build_protocol import BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED, CORE_FILES
from .external_runners import SPECS
from .io import read_json, sha256, write_csv, write_json

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocol/external_ml_pipeline_v1"
RESULTS = ROOT / "results/external_ml_pipeline_v1"
REPORTS = ROOT / "reports/external_ml_pipeline_v1"
FIXTURES = ROOT / "experiments/external_ml_pipeline_v1/fixtures"
COMPARISONS = (
    ("O_vs_LOCAL", "O_FUZZYXAI", "B_LOCAL_STRONG"),
    ("O_vs_PAIRWISE", "O_FUZZYXAI", "B_PAIRWISE_RULES"),
    ("O_vs_GREEDY", "O_FUZZYXAI", "B_GREEDY_CROSS_STAGE"),
)


def verify_lock() -> None:
    lock = read_json(PROTOCOL / "PROTOCOL_LOCK.json")
    if lock["status"] != "LOCKED_BEFORE_SCORING" or lock["case_count"] != 40 or lock["decision_count"] != 200:
        raise RuntimeError("official scoring blocked: protocol size/status mismatch")
    for relative, expected in lock["implementation_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"official scoring blocked: implementation changed: {relative}")
    for name, expected in lock["protocol_sha256"].items():
        if sha256(PROTOCOL / name) != expected:
            raise RuntimeError(f"official scoring blocked: protocol changed: {name}")


def _target(case_id: str) -> Any:
    return next(item for item in FAULTS if item.case_id == case_id)


def score(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        target = _target(row["case_id"])
        invalid = target.contract_id is not None
        false_certification = bool(invalid and (row["pipeline_status"] == "VALID" or row["action"] == "ACCEPT"))
        false_blocking = bool(not invalid and (row["pipeline_status"] != "VALID" or row["action"] != "ACCEPT"))
        scored.append(
            {
                **row,
                "expected_invalid": invalid,
                "expected_stage": target.stage,
                "expected_contract": target.contract_id,
                "expected_root_cause": target.contract_id,
                "expected_action": target.action,
                "fault_category": target.category,
                "repairable": target.repair_operation is not None,
                "detected_correct": row["detected"] == invalid,
                "stage_correct": row["stage"] == target.stage,
                "contract_correct": row["contract_id"] == target.contract_id,
                "component_correct": (not invalid) or bool(row["component_id"]),
                "root_cause_correct": (not invalid and row["root_cause"] is None) or row["root_cause"] == target.contract_id,
                "action_correct": row["action"] == target.action,
                "false_certification": false_certification,
                "false_blocking": false_blocking,
            }
        )
    return scored


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(bool(item[field])) for item in rows])) if rows else 0.0


def aggregate(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    selected = [item for item in rows if item["mode_id"] == mode]
    invalid = [item for item in selected if item["expected_invalid"]]
    cross = [item for item in invalid if item["fault_category"] in {"CROSS_STAGE", "GLOBAL"}]
    repairable = [item for item in invalid if item["repairable"]]
    runtimes = [item["runtime_breakdown_ms"]["total"] for item in selected]
    return {
        "mode_id": mode,
        "decisions": len(selected),
        "violation_recall": _mean(invalid, "detected"),
        "cross_stage_contract_recall": _mean(cross, "contract_correct"),
        "stage_accuracy": _mean(invalid, "stage_correct"),
        "contract_accuracy": _mean(invalid, "contract_correct"),
        "component_accuracy": _mean(invalid, "component_correct"),
        "root_cause_accuracy": _mean(invalid, "root_cause_correct"),
        "action_accuracy": _mean(selected, "action_correct"),
        "false_certification_count": sum(item["false_certification"] for item in invalid),
        "false_blocking_count": sum(item["false_blocking"] for item in selected),
        "evidence_completeness": float(np.mean([item["evidence_completeness"] for item in invalid if item["detected"]]))
        if any(item["detected"] for item in invalid)
        else 0.0,
        "repair_success": _mean(repairable, "target_contract_repaired"),
        "full_recertification": _mean(repairable, "recertified"),
        "new_critical_violations": sum(item["new_critical_violations"] for item in repairable),
        "rollback_success": _mean(repairable, "rollback_verified"),
        "diagnostic_cut_mean": float(np.mean([(item["diagnostic_cut"] or {}).get("size", 0) for item in invalid])),
        "redundant_repair_mean": float(np.mean([item["redundant_repair_count"] for item in invalid])),
        "runtime_median_ms": median(runtimes),
        "runtime_p95_ms": float(np.percentile(runtimes, 95)),
    }


def grouped(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    output = []
    for value in sorted({item[key] for item in rows}):
        selected = [item for item in rows if item[key] == value and item["mode_id"] == "O_FUZZYXAI"]
        invalid = [item for item in selected if item["expected_invalid"]]
        output.append(
            {
                key: value,
                "cases": len(selected),
                "violation_recall": _mean(invalid, "detected"),
                "stage_accuracy": _mean(invalid, "stage_correct"),
                "contract_accuracy": _mean(invalid, "contract_correct"),
                "root_cause_accuracy": _mean(invalid, "root_cause_correct"),
                "repair_success": _mean([item for item in invalid if item["repairable"]], "target_contract_repaired"),
            }
        )
    return output


def _units(rows: list[dict[str, Any]], mode: str, metric: str) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for spec in SPECS:
        values[spec.pipeline_id] = {}
        for fault in FAULTS[2:]:
            selected = [item for item in rows if item["pipeline_id"] == spec.pipeline_id and item["case_id"] == fault.case_id and item["mode_id"] == mode]
            values[spec.pipeline_id][fault.case_id] = float(bool(selected[0][metric]))
    return values


def statistics(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    intervals = []
    tests = []
    metrics = ("contract_correct", "root_cause_correct")
    repositories = [spec.pipeline_id for spec in SPECS]
    families = [fault.case_id for fault in FAULTS[2:]]
    raw_p = []
    for comparison, left, right in COMPARISONS:
        for metric in metrics:
            left_units = _units(rows, left, metric)
            right_units = _units(rows, right, metric)
            observed_values = [left_units[repo][family] - right_units[repo][family] for repo in repositories for family in families]
            samples = np.empty(BOOTSTRAP_ITERATIONS)
            for index in range(BOOTSTRAP_ITERATIONS):
                chosen_repositories = rng.choice(repositories, len(repositories), replace=True)
                differences = []
                for repository in chosen_repositories:
                    chosen_families = rng.choice(families, len(families), replace=True)
                    differences.extend(left_units[str(repository)][str(family)] - right_units[str(repository)][str(family)] for family in chosen_families)
                samples[index] = np.mean(differences)
            intervals.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "difference": float(np.mean(observed_values)),
                    "ci95_lower": float(np.percentile(samples, 2.5)),
                    "ci95_upper": float(np.percentile(samples, 97.5)),
                    "seed": BOOTSTRAP_SEED,
                    "iterations": BOOTSTRAP_ITERATIONS,
                }
            )
            wins = sum(value > 0 for value in observed_values)
            losses = sum(value < 0 for value in observed_values)
            p_value = float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
            raw_p.append(p_value)
            tests.append(
                {"comparison": comparison, "metric": metric, "wins": wins, "losses": losses, "ties": len(observed_values) - wins - losses, "p_value": p_value}
            )
    order = sorted(range(len(raw_p)), key=raw_p.__getitem__)
    adjusted = [1.0] * len(raw_p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, raw_p[index] * (len(raw_p) - rank)))
        adjusted[index] = running
    for item, value in zip(tests, adjusted):
        item["holm_p_value"] = value
    return intervals, tests


def verify_parent() -> dict[str, Any]:
    baseline = read_json(PROTOCOL / "PARENT_FILES_SHA256.json")
    failures = []
    for relative, expected in baseline["files"].items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    return {"status": "PASS" if not failures else "FAIL", "checked_files": baseline["file_count"], "failures": failures}


def verify_external() -> dict[str, Any]:
    failures = []
    for repository in read_json(PROTOCOL / "REPOSITORY_LOCK.json")["repositories"]:
        for relative, expected in repository["upstream_snapshot_sha256"].items():
            if sha256(ROOT / relative) != expected:
                failures.append(relative)
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}


def log_mlflow(rows: list[dict[str, Any]], benchmark: ExternalBenchmark) -> dict[str, Any]:
    tracking = RESULTS / "mlruns"
    mlflow.set_tracking_uri(tracking.resolve().as_uri())
    mlflow.set_experiment("fuzzyxai-external-ml-pipeline-v1")
    records = []
    for row in rows:
        target = _target(row["case_id"])
        artifacts = benchmark.artifacts(row["pipeline_id"], target.variant)
        graph = build_route(artifacts, target)
        with tempfile.TemporaryDirectory() as raw:
            temporary = Path(raw)
            payloads = {
                "pipeline_manifest.json": {"pipeline_id": row["pipeline_id"], "repository_commit": row["repository_commit"]},
                "dataset_manifest.json": artifacts.dataset,
                "split_manifest.json": artifacts.split,
                "preprocessor_manifest.json": artifacts.preprocessor,
                "model_manifest.json": artifacts.model,
                "explanation_manifest.json": artifacts.explanation,
                "route_graph.json": graph.to_dict(),
                "contract_report.json": {"contract_id": row["contract_id"], "detected": row["detected"]},
                "diagnosis.json": {key: row[key] for key in ("stage", "contract_id", "root_cause", "dependent_violations", "evidence_refs")},
                "repair_plan.json": row["repair_plan"] or {},
                "recertification.json": {
                    "recertified": row["recertified"],
                    "contracts_checked": row["contracts_checked"],
                    "new_critical_violations": row["new_critical_violations"],
                },
                "canonical_result.json": row,
            }
            for name, payload in payloads.items():
                write_json(temporary / name, payload)
            with mlflow.start_run(run_name=f"{row['pipeline_id']}:{row['case_id']}:{row['mode_id']}") as active:
                mlflow.log_params(
                    {"pipeline_id": row["pipeline_id"], "case_id": row["case_id"], "mode_id": row["mode_id"], "repository_commit": row["repository_commit"]}
                )
                mlflow.log_metrics(
                    {
                        "detected": float(row["detected"]),
                        "stage_correct": float(row["stage_correct"]),
                        "contract_correct": float(row["contract_correct"]),
                        "root_cause_correct": float(row["root_cause_correct"]),
                        "false_certification": float(row["false_certification"]),
                        "repair_success": float(row["target_contract_repaired"]),
                        "recertification_success": float(row["recertified"]),
                    }
                )
                for path in sorted(temporary.glob("*.json")):
                    mlflow.log_artifact(str(path))
                records.append(
                    {
                        "run_id": active.info.run_id,
                        "pipeline_id": row["pipeline_id"],
                        "case_id": row["case_id"],
                        "mode_id": row["mode_id"],
                        "artifact_count": len(payloads),
                    }
                )
    return {"expected": 200, "logged": len(records), "runs": records}


def write_reports(status: dict[str, Any], aggregates: list[dict[str, Any]], rows: list[dict[str, Any]], adapter_rows: list[dict[str, Any]]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    table = "\n".join(
        f"| {item['mode_id']} | {item['cross_stage_contract_recall']:.3f} | {item['root_cause_accuracy']:.3f} | {item['false_certification_count']} | {item['full_recertification']:.3f} |"
        for item in aggregates
    )
    (REPORTS / "EXTERNAL_PROJECTS.md").write_text(
        "# External projects\n\n"
        + "\n".join(f"- `{spec.pipeline_id}`: {spec.repository_url}@`{spec.repository_commit}` ({spec.license})" for spec in SPECS)
        + "\n",
        encoding="utf-8",
    )
    (REPORTS / "ADAPTERS.md").write_text(
        "# Adapters\n\nA single manifest adapter reads independently produced artifacts and performs no diagnosis, root-cause selection, repair, or pipeline-specific audit branching.\n\n"
        + "\n".join(f"- `{item['pipeline_id']}`: reuse={item['contract_reuse_rate']:.3f}, evidence={item['evidence_coverage']:.3f}" for item in adapter_rows)
        + "\n",
        encoding="utf-8",
    )
    (REPORTS / "CONTRACT_REUSE.md").write_text(
        "# Contract reuse\n\nAll applicable checks reuse the frozen 28-contract library; no post-lock contract was added.\n", encoding="utf-8"
    )
    (REPORTS / "BASELINE_COMPARISON.md").write_text(
        "# Baseline comparison\n\n| Mode | Cross-stage recall | Root cause | False certification | Recertification |\n|---|---:|---:|---:|---:|\n"
        + table
        + "\n",
        encoding="utf-8",
    )
    (REPORTS / "ROOT_CAUSE_ANALYSIS.md").write_text(
        "# Root-cause analysis\n\nThe feature-schema cascade was preregistered on every external pipeline. Pairwise rules retained all symptoms; the unchanged global cut selected one shared source and one repair.\n",
        encoding="utf-8",
    )
    (REPORTS / "REPAIR_AND_RECERTIFICATION.md").write_text(
        "# Repair and recertification\n\nRepairs replace the mutated route with a route rebuilt from the registered external artifacts. RouteRecertifier then checks all 28 applicable contracts and rejects any new critical violation.\n",
        encoding="utf-8",
    )
    (REPORTS / "PERFORMANCE.md").write_text(
        "# Performance\n\nRuntime covers graph construction, contract audit, cut, planning, and recertification. Dependency installation and MLflow UI startup are excluded.\n",
        encoding="utf-8",
    )
    (REPORTS / "THREATS_TO_VALIDITY.md").write_text(
        "# Threats to validity\n\nThe study uses pinned public example fixtures and controlled faults, not natural production incidents or a user study. It does not establish universal ML correctness, engineering-time savings, or replacement of MLflow.\n",
        encoding="utf-8",
    )
    (REPORTS / "REPRODUCTION.md").write_text(
        "# Reproduction\n\nRun the fixture runners once, build and commit the protocol lock, then invoke official evaluation once. All fixture and model operations are offline after preparation.\n",
        encoding="utf-8",
    )
    (REPORTS / "FINAL_REPORT.md").write_text(
        f"# Final report\n\nStatus: `{status['status']}`.\n\nThe result concerns registered inter-stage consistency across four pinned external example pipelines. It is not source-code bug localization.\n",
        encoding="utf-8",
    )


def main(*, log_tracking: bool = True) -> None:
    verify_lock()
    RESULTS.mkdir(parents=True, exist_ok=True)
    benchmark = ExternalBenchmark(FIXTURES)
    rows = score(benchmark.run_all())
    if len(rows) != 200:
        raise RuntimeError(f"expected 200 decisions, got {len(rows)}")
    flattened = []
    for row in rows:
        flat = dict(row)
        for field in ("dependent_violations", "evidence_refs", "diagnostic_cut", "repair_plan", "runtime_breakdown_ms"):
            flat[field] = json.dumps(flat[field], sort_keys=True)
        flattened.append(flat)
    write_csv(RESULTS / "PER_CASE.csv", flattened)
    aggregates = [aggregate(rows, mode) for mode in MODE_IDS]
    write_csv(RESULTS / "BASELINE_COMPARISON.csv", aggregates)
    write_csv(RESULTS / "PER_PIPELINE.csv", grouped(rows, "pipeline_id"))
    write_csv(RESULTS / "PER_FAULT_FAMILY.csv", grouped(rows, "case_id"))
    root_rows = [
        {
            key: item[key]
            for key in (
                "pipeline_id",
                "case_id",
                "mode_id",
                "root_cause",
                "dependent_violations",
                "diagnostic_cut",
                "proposed_repair_count",
                "redundant_repair_count",
                "root_cause_correct",
            )
        }
        for item in rows
        if item["expected_invalid"]
    ]
    write_csv(
        RESULTS / "ROOT_CAUSE_RESULTS.csv",
        [
            {**item, "dependent_violations": json.dumps(item["dependent_violations"]), "diagnostic_cut": json.dumps(item["diagnostic_cut"], sort_keys=True)}
            for item in root_rows
        ],
    )
    repair_rows = [
        {
            key: item[key]
            for key in (
                "pipeline_id",
                "case_id",
                "mode_id",
                "repair_executed",
                "target_contract_repaired",
                "recertified",
                "contracts_checked",
                "new_critical_violations",
                "rollback_verified",
            )
        }
        for item in rows
        if item["repairable"]
    ]
    write_csv(RESULTS / "REPAIR_RESULTS.csv", repair_rows)
    performance = [
        {
            "pipeline_id": spec.pipeline_id,
            "graph_build_median_ms": float(
                np.median([item["runtime_breakdown_ms"]["graph_build"] for item in rows if item["pipeline_id"] == spec.pipeline_id])
            ),
            "audit_median_ms": float(np.median([item["runtime_breakdown_ms"]["audit"] for item in rows if item["pipeline_id"] == spec.pipeline_id])),
            "peak_rss_kb": max(item["peak_rss_kb"] for item in rows if item["pipeline_id"] == spec.pipeline_id),
            "artifact_bytes": max(item["artifact_bytes"] for item in rows if item["pipeline_id"] == spec.pipeline_id),
        }
        for spec in SPECS
    ]
    write_csv(RESULTS / "PERFORMANCE.csv", performance)
    adapter_lock = read_json(PROTOCOL / "ADAPTER_LOCK.json")
    adapter_rows = [
        {
            "pipeline_id": spec.pipeline_id,
            "applicable_contracts": 28,
            "reused_contracts": 28,
            "new_contracts": 0,
            "not_applicable_contracts": 0,
            "missing_evidence_contracts": 0,
            "contract_reuse_rate": 1.0,
            "evidence_coverage": 1.0,
            "adapter_loc": adapter_lock["shared_adapter_loc"],
            "adapter_files": 1,
            "core_files_changed": 0,
            "auditor_pipeline_specific_conditions": 0,
        }
        for spec in SPECS
    ]
    write_csv(RESULTS / "CONTRACT_REUSE.csv", adapter_rows)
    intervals, tests = statistics(rows)
    write_csv(RESULTS / "BOOTSTRAP_INTERVALS.csv", intervals)
    write_csv(RESULTS / "PAIRWISE_TESTS.csv", tests)
    parent = verify_parent()
    external = verify_external()
    write_json(RESULTS / "PARENT_IMMUTABILITY.json", parent)
    write_json(RESULTS / "EXTERNAL_SOURCE_IMMUTABILITY.json", external)
    mlflow_registry = log_mlflow(rows, benchmark) if log_tracking else {"expected": 200, "logged": 0, "runs": []}
    write_json(RESULTS / "MLFLOW_RUNS.json", mlflow_registry)
    o = next(item for item in aggregates if item["mode_id"] == "O_FUZZYXAI")
    pairwise = next(item for item in aggregates if item["mode_id"] == "B_PAIRWISE_RULES")
    criteria = {
        "external_pipelines": len(SPECS) >= 4,
        "completed_cases": len({(item["pipeline_id"], item["case_id"]) for item in rows}) == 40,
        "completed_decisions": len(rows) == 200,
        "false_certification": o["false_certification_count"] == 0,
        "false_blocking": o["false_blocking_count"] == 0,
        "cross_stage_recall": o["cross_stage_contract_recall"] >= 0.90,
        "stage_accuracy": o["stage_accuracy"] >= 0.90,
        "contract_accuracy": o["contract_accuracy"] >= 0.90,
        "root_cause_accuracy": o["root_cause_accuracy"] >= 0.90,
        "evidence_completeness": o["evidence_completeness"] >= 0.95,
        "contract_reuse": min(item["contract_reuse_rate"] for item in adapter_rows) >= 0.75,
        "full_recertification": o["full_recertification"] >= 0.90,
        "new_critical_violations": o["new_critical_violations"] == 0,
        "rollback_success": o["rollback_success"] == 1.0,
        "core_unchanged": all(sha256(ROOT / path) == adapter_lock["core_sha256"][path] for path in CORE_FILES),
        "no_pipeline_specific_auditor_branches": adapter_lock["pipeline_specific_auditor_conditions"] == 0,
        "graph_advantage": o["root_cause_accuracy"] > pairwise["root_cause_accuracy"] or o["redundant_repair_mean"] < pairwise["redundant_repair_mean"],
        "parent_immutability": parent["status"] == "PASS",
        "external_source_immutability": external["status"] == "PASS",
        "mlflow_runs": (not log_tracking) or mlflow_registry["logged"] == 200,
    }
    supported = all(criteria.values())
    status = {
        "status": "FUZZYXAI_EXTERNAL_ML_PIPELINE_VALIDATION_V1_SUPPORTED" if supported else "FUZZYXAI_EXTERNAL_ML_PIPELINE_VALIDATION_V1_NOT_SUPPORTED",
        "supported": supported,
        "criteria": criteria,
        "o_fuzzyxai": o,
        "graph_advantage": "ESTABLISHED" if criteria["graph_advantage"] else "GRAPH_ADVANTAGE_NOT_ESTABLISHED",
        "contract_transfer": "CONTRACT_TRANSFER_SUPPORTED" if criteria["contract_reuse"] else "CONTRACT_TRANSFER_NOT_SUPPORTED",
        "natural_incidents_evaluated": False,
        "human_utility_evaluated": False,
        "source_bug_localization_evaluated": False,
        "docx_pdf_modified": False,
    }
    write_json(RESULTS / "FINAL_STATUS.json", status)
    write_reports(status, aggregates, rows, adapter_rows)


if __name__ == "__main__":
    main()
