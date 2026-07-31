#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from fuzzyxai.ml_vertical.comparative import MODE_IDS, evaluate_mode, project_mode_input
from fuzzyxai.ml_vertical.pipeline import ALL_SCENARIOS, CONTRACT_STAGE, MLPipelineService

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocol/ml_pipeline_v2_comparative"
RESULTS = ROOT / "results/ml_pipeline_v2_comparative"
REPORTS = ROOT / "reports/ml_pipeline_v2_comparative"
BOOTSTRAP_SEED = 1729
BOOTSTRAP_ITERATIONS = 10_000
BINARY_METRICS = ("detected_correct", "stage_correct", "contract_correct", "action_correct")
COMPARISONS = (
    ("O_vs_B2", "A4", "B2"),
    ("A2_vs_A1", "A2", "A1"),
    ("A3_vs_A2", "A3", "A2"),
    ("A4_vs_A3", "A4", "A3"),
    ("O_vs_B1", "A4", "B1"),
)
CROSS_STAGE_CASES = (
    "S11_TARGET_LEAKAGE",
    "S12_SPLIT_OVERLAP",
    "S13_PREPROCESSOR_FULL_FIT",
    "S16_MODEL_ARTIFACT_TAMPER",
    "S2_EXPLAINER_VERSION_MISMATCH",
    "S17_SHAP_INCONSISTENCY",
)


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def repository_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_protocol_lock() -> dict[str, Any]:
    required = (
        "COMPARATIVE_PROTOCOL_LOCK.json",
        "MODE_DEFINITIONS.json",
        "PRIMARY_COMPARISONS.json",
        "METRIC_DEFINITIONS.json",
        "ACCEPTANCE_CRITERIA.json",
        "SCORING_TARGETS.json",
    )
    missing = [name for name in required if not (PROTOCOL / name).is_file()]
    if missing:
        raise RuntimeError(f"comparative scoring blocked; missing protocol lock files: {missing}")
    lock = read_json(PROTOCOL / "COMPARATIVE_PROTOCOL_LOCK.json")
    for relative, expected in lock["locked_file_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"comparative scoring blocked; lock mismatch: {relative}")
    if lock["bootstrap_seed"] != BOOTSTRAP_SEED or lock["bootstrap_iterations"] != BOOTSTRAP_ITERATIONS:
        raise RuntimeError("comparative scoring blocked; statistical constants differ from protocol lock")
    return lock


def score_row(payload: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    expected_detected = target["contract_id"] is not None
    false_certification = bool(
        expected_detected
        and not payload["recertified"]
        and (payload["pipeline_status"] == "VALID" or payload["action"] == "ACCEPT")
    )
    return {
        **payload,
        "expected_detected": expected_detected,
        "expected_stage": target["stage"],
        "expected_contract_id": target["contract_id"],
        "expected_component_id": target["component_id"],
        "expected_action": target["action"],
        "repair_expected": target["repair_expected"],
        "detected_correct": payload["detected"] == expected_detected,
        "stage_correct": payload["stage"] == target["stage"],
        "contract_correct": payload["contract_id"] == target["contract_id"],
        "component_correct": payload["component_id"] == target["component_id"],
        "action_correct": payload["action"] == target["action"],
        "false_certification": false_certification,
        "repair_success": bool(
            target["repair_expected"]
            and payload["repair_executed"]
            and payload["target_contract_repaired"]
        ),
        "recertification_success": bool(target["repair_expected"] and payload["recertified"]),
    }


def aggregate_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    repair_rows = [row for row in rows if row["repair_expected"]]
    runtimes = [float(row["runtime_ms"]) for row in rows]
    return {
        "mode_id": rows[0]["mode_id"],
        "role": "O_FULL_FUZZYXAI" if rows[0]["mode_id"] == "A4" else rows[0]["mode_id"],
        "scenario_count": len(rows),
        "violation_detection_accuracy": _mean(rows, "detected_correct"),
        "stage_localization_accuracy": _mean(rows, "stage_correct"),
        "contract_identification_accuracy": _mean(rows, "contract_correct"),
        "component_localization_accuracy": _mean(rows, "component_correct"),
        "action_accuracy": _mean(rows, "action_correct"),
        "false_certification_rate": _mean(rows, "false_certification"),
        "false_certification_count": sum(bool(row["false_certification"]) for row in rows),
        "abstention_rate": _mean(rows, "abstained"),
        "evidence_completeness": float(np.mean([float(row["evidence_completeness"]) for row in rows])),
        "repair_success": _mean(repair_rows, "repair_success"),
        "repair_success_count": sum(bool(row["repair_success"]) for row in repair_rows),
        "full_recertification_success": _mean(repair_rows, "recertification_success"),
        "full_recertification_count": sum(bool(row["recertification_success"]) for row in repair_rows),
        "new_critical_violations_after_repair": sum(
            int(row["new_critical_violations"] or 0) for row in repair_rows if row["repair_executed"]
        ),
        "runtime_median_ms": median(runtimes),
        "runtime_p95_ms": float(np.percentile(runtimes, 95)),
        "diagnostic_cut_count": sum(row["diagnostic_cut"] is not None for row in rows),
    }


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([bool(row[field]) for row in rows])) if rows else 0.0


def mcnemar_exact(left: list[bool], right: list[bool]) -> tuple[int, int, float]:
    left_only = sum(a and not b for a, b in zip(left, right, strict=True))
    right_only = sum(b and not a for a, b in zip(left, right, strict=True))
    discordant = left_only + right_only
    if discordant == 0:
        return left_only, right_only, 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1))
    return left_only, right_only, min(1.0, 2.0 * tail / (2**discordant))


def paired_bootstrap(left: list[bool], right: list[bool], *, seed: int, iterations: int) -> tuple[float, float, float]:
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        selected = rng.integers(0, len(differences), size=len(differences))
        samples[index] = float(np.mean(differences[selected]))
    return float(np.mean(differences)), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda pair: float(pair[1]["p_value_raw"]))
    running = 0.0
    count = len(rows)
    adjusted: dict[int, float] = {}
    for rank, (original_index, row) in enumerate(ordered):
        value = min(1.0, (count - rank) * float(row["p_value_raw"]))
        running = max(running, value)
        adjusted[original_index] = running
    for index, row in enumerate(rows):
        row["p_value_holm"] = adjusted[index]


def statistical_tables(rows_by_mode: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tests: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    for comparison_id, left_mode, right_mode in COMPARISONS:
        left_by_id = {row["scenario_id"]: row for row in rows_by_mode[left_mode]}
        right_by_id = {row["scenario_id"]: row for row in rows_by_mode[right_mode]}
        scenario_ids = list(ALL_SCENARIOS)
        for metric in BINARY_METRICS:
            left = [bool(left_by_id[item][metric]) for item in scenario_ids]
            right = [bool(right_by_id[item][metric]) for item in scenario_ids]
            left_only, right_only, p_value = mcnemar_exact(left, right)
            tests.append(
                {
                    "comparison_id": comparison_id,
                    "left_mode": left_mode,
                    "right_mode": right_mode,
                    "metric": metric,
                    "left_correct_right_incorrect": left_only,
                    "left_incorrect_right_correct": right_only,
                    "p_value_raw": p_value,
                    "p_value_holm": 0.0,
                }
            )
            difference, lower, upper = paired_bootstrap(
                left,
                right,
                seed=BOOTSTRAP_SEED,
                iterations=BOOTSTRAP_ITERATIONS,
            )
            intervals.append(
                {
                    "comparison_id": comparison_id,
                    "left_mode": left_mode,
                    "right_mode": right_mode,
                    "metric": metric,
                    "difference": difference,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                }
            )
    holm_adjust(tests)
    return tests, intervals


def log_comparative_run(
    row: dict[str, Any],
    *,
    tracking_uri: str,
    git_commit: str,
    scenario_sha256: str,
    mode_sha256: str,
) -> dict[str, Any]:
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("comparative evaluation requires the locked MLflow dependency") from exc
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fuzzyxai-ml-pipeline-v2-comparative")
    with mlflow.start_run(run_name=f"{row['scenario_id']}-{row['mode_id']}") as active:
        mlflow.log_params(
            {
                "scenario_id": row["scenario_id"],
                "mode_id": row["mode_id"],
                "pipeline_version": "2.0.0",
                "git_commit": git_commit,
                "scenario_sha256": scenario_sha256,
                "mode_sha256": mode_sha256,
            }
        )
        mlflow.log_metrics(
            {
                "detected": float(row["detected"]),
                "stage_correct": float(row["stage_correct"]),
                "contract_correct": float(row["contract_correct"]),
                "component_correct": float(row["component_correct"]),
                "action_correct": float(row["action_correct"]),
                "false_certification": float(row["false_certification"]),
                "evidence_completeness": float(row["evidence_completeness"]),
                "repair_success": float(row["repair_success"]),
                "recertification_success": float(row["recertification_success"]),
                "runtime_ms": float(row["runtime_ms"]),
            }
        )
        with tempfile.TemporaryDirectory(prefix="fuzzyxai-comparative-") as temporary:
            root = Path(temporary)
            public_result = {key: value for key, value in row.items() if not key.startswith("expected_")}
            write_json(root / "comparative_result.json", public_result)
            write_json(
                root / "mode_input_manifest.json",
                {
                    "scenario_id": row["scenario_id"],
                    "mode_id": row["mode_id"],
                    "scenario_sha256": scenario_sha256,
                    "mode_sha256": mode_sha256,
                    "gold_fields_logged": False,
                },
            )
            mlflow.log_artifacts(str(root), artifact_path="comparative")
        return {"scenario_id": row["scenario_id"], "mode_id": row["mode_id"], "run_id": active.info.run_id, "artifact_uri": active.info.artifact_uri}


def render_reports(
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    status: dict[str, Any],
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    table = [
        "| Mode | Detect | Stage | Contract | Component | Action | False cert | Evidence | Repair | Recert |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        table.append(
            f"| {item['mode_id']} | {item['violation_detection_accuracy']:.3f} | "
            f"{item['stage_localization_accuracy']:.3f} | {item['contract_identification_accuracy']:.3f} | "
            f"{item['component_localization_accuracy']:.3f} | {item['action_accuracy']:.3f} | "
            f"{item['false_certification_rate']:.3f} | {item['evidence_completeness']:.3f} | "
            f"{item['repair_success']:.3f} | {item['full_recertification_success']:.3f} |"
        )
    (REPORTS / "COMPARATIVE_REPORT.md").write_text(
        "# ML Pipeline v2 comparative evaluation\n\n"
        + "\n".join(table)
        + f"\n\nFinal status: `{status['status']}`. A4 is the full FuzzyXAI mode O. "
        "MLflow is treated as run registration rather than a competing diagnostic system.\n",
        encoding="utf-8",
    )
    ablation = {item["mode_id"]: item for item in aggregates}
    (REPORTS / "ABLATION_REPORT.md").write_text(
        "# Ablation report\n\n"
        f"A1 contract accuracy: `{ablation['A1']['contract_identification_accuracy']:.4f}`.\n\n"
        f"A2 contract accuracy: `{ablation['A2']['contract_identification_accuracy']:.4f}`.\n\n"
        f"A3 diagnostic cuts: `{ablation['A3']['diagnostic_cut_count']}/18`.\n\n"
        f"A4 full recertifications: `{ablation['A4']['full_recertification_count']}/5`.\n",
        encoding="utf-8",
    )
    by_pair = {(row["scenario_id"], row["mode_id"]): row for row in rows}
    case_lines = ["# Six cross-stage cases", ""]
    for scenario_id in CROSS_STAGE_CASES:
        b0 = by_pair[(scenario_id, "B0")]
        b1 = by_pair[(scenario_id, "B1")]
        b2 = by_pair[(scenario_id, "B2")]
        full = by_pair[(scenario_id, "A4")]
        case_lines.extend(
            [
                f"## {scenario_id}",
                "",
                f"- B0: detected `{b0['detected']}`; available standard log fields do not include RouteGraph contracts.",
                f"- B1: detected `{b1['detected']}`; MLflow preserves registered run data without inferring a diagnosis.",
                f"- B2: detected `{b2['detected']}`; contract `{b2['contract_id']}`.",
                f"- O: `{full['stage']}` / `{full['contract_id']}` / `{full['component_id']}`; action `{full['action']}`.",
                f"- Repair and recertification: `{full['repair_executed']}` / `{full['recertified']}`.",
                "",
            ]
        )
    (REPORTS / "SIX_CROSS_STAGE_CASES.md").write_text("\n".join(case_lines), encoding="utf-8")
    (REPORTS / "LIMITATIONS.md").write_text(
        "# Limitations\n\n"
        "The evaluation uses eighteen fixed registered mutations. It does not establish detection of arbitrary ML errors, "
        "human-time reduction, user utility, or transfer to natural software incidents. MLflow and FuzzyXAI have different roles. "
        "P-values are secondary because the scenario count is small.\n",
        encoding="utf-8",
    )
    (REPORTS / "REPRODUCTION.md").write_text(
        "# Reproduction\n\n"
        "The protocol lock must be committed before scoring.\n\n"
        "```bash\n"
        "PYTHONPATH=framework/fuzzyxai:. python scripts/ml_pipeline_v2_comparative/run_evaluation.py\n"
        "make ml-pipeline-v2-comparative-test\n"
        "```\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=(RESULTS / "mlruns").as_uri())
    parser.add_argument("--skip-mlflow", action="store_true")
    args = parser.parse_args()
    lock = verify_protocol_lock()
    mode_definitions = read_json(PROTOCOL / "MODE_DEFINITIONS.json")["modes"]
    scenario_lock = read_json(ROOT / "protocol/ml_pipeline_v2/SCENARIO_LOCK.json")
    scenario_hashes = {
        scenario_id: canonical_sha256(
            scenario_lock["v2_scenarios"].get(scenario_id, {"frozen_v1_scenario_id": scenario_id})
        )
        for scenario_id in ALL_SCENARIOS
    }

    service = MLPipelineService()
    unscored: list[dict[str, Any]] = []
    for scenario_id in ALL_SCENARIOS:
        run = service.execute_scenario(scenario_id)
        for mode_id in MODE_IDS:
            mode_input = project_mode_input(run, mode_id)
            unscored.append(asdict(evaluate_mode(mode_input)))

    # Gold is physically read only after all 162 mode decisions are complete.
    targets = read_json(PROTOCOL / "SCORING_TARGETS.json")["targets"]
    target_by_id = {item["scenario_id"]: item for item in targets}
    rows = [score_row(item, target_by_id[item["scenario_id"]]) for item in unscored]
    rows_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_mode[row["mode_id"]].append(row)
    aggregates = [aggregate_mode(rows_by_mode[mode_id]) for mode_id in MODE_IDS]
    pairwise, intervals = statistical_tables(rows_by_mode)

    aggregate_by_mode = {item["mode_id"]: item for item in aggregates}
    full = aggregate_by_mode["A4"]
    cross_advantage = sum(
        by_o["contract_correct"] and not by_b2["contract_correct"]
        for by_o, by_b2 in (
            (
                next(row for row in rows_by_mode["A4"] if row["scenario_id"] == scenario_id),
                next(row for row in rows_by_mode["B2"] if row["scenario_id"] == scenario_id),
            )
            for scenario_id in CROSS_STAGE_CASES
        )
    )
    gates = {
        "fuzzyxai_false_certification_zero": full["false_certification_count"] == 0,
        "fuzzyxai_stage_accuracy_one": full["stage_localization_accuracy"] == 1.0,
        "fuzzyxai_contract_accuracy_one": full["contract_identification_accuracy"] == 1.0,
        "fuzzyxai_action_accuracy_one": full["action_accuracy"] == 1.0,
        "fuzzyxai_repair_recertification_five_of_five": full["repair_success_count"] == 5 and full["full_recertification_count"] == 5,
        "fuzzyxai_evidence_completeness_one": full["evidence_completeness"] == 1.0,
        "cross_stage_advantage_at_least_two": cross_advantage >= 2,
        "a2_contract_accuracy_exceeds_a1": aggregate_by_mode["A2"]["contract_identification_accuracy"] > aggregate_by_mode["A1"]["contract_identification_accuracy"],
        "a4_recertification_exceeds_a3": aggregate_by_mode["A4"]["full_recertification_success"] > aggregate_by_mode["A3"]["full_recertification_success"],
        "all_a4_repairs_recheck_28_contracts": all(
            row["contracts_rechecked_count"] == len(CONTRACT_STAGE)
            for row in rows_by_mode["A4"]
            if row["repair_expected"]
        ),
    }
    supported = all(gates.values())
    status = {
        "status": "FUZZYXAI_ML_PIPELINE_V2_COMPARATIVE_SUPPORTED" if supported else "COMPARATIVE_ADVANTAGE_NOT_ESTABLISHED",
        "evaluation_status": "FUZZYXAI_ML_PIPELINE_V2_COMPARATIVE_EVALUATION",
        "supported": supported,
        "scenario_count": len(ALL_SCENARIOS),
        "mode_count": len(MODE_IDS),
        "mode_decision_count": len(rows),
        "mlflow_run_count": 0 if args.skip_mlflow else len(rows),
        "cross_stage_advantage_count": cross_advantage,
        "gates": gates,
        "protocol_lock_sha256": hashlib.sha256((PROTOCOL / "COMPARATIVE_PROTOCOL_LOCK.json").read_bytes()).hexdigest(),
        "implementation_commit": lock["implementation_commit"],
        "scoring_commit": repository_head(),
        "parent_pipeline_status": "FUZZYXAI_ML_PIPELINE_V2_IMPLEMENTED",
        "parent_results_modified": False,
        "docx_pdf_modified": False,
        "claims": {
            "arbitrary_ml_error_detection": False,
            "replaces_mlflow_or_kubeflow": False,
            "human_time_reduction": False,
            "user_utility": False,
            "natural_incident_transfer": False,
        },
    }

    mlflow_runs: list[dict[str, Any]] = []
    if not args.skip_mlflow:
        git_commit = repository_head()
        for row in rows:
            mlflow_runs.append(
                log_comparative_run(
                    row,
                    tracking_uri=args.tracking_uri,
                    git_commit=git_commit,
                    scenario_sha256=scenario_hashes[row["scenario_id"]],
                    mode_sha256=canonical_sha256(mode_definitions[row["mode_id"]]),
                )
            )

    write_jsonl(RESULTS / "PER_SCENARIO_MODE.jsonl", rows)
    write_csv(RESULTS / "AGGREGATES.csv", aggregates)
    write_csv(
        RESULTS / "ABLATION_RESULTS.csv",
        [item for item in aggregates if item["mode_id"].startswith("A")],
    )
    write_csv(RESULTS / "PAIRWISE_TESTS.csv", pairwise)
    write_csv(RESULTS / "BOOTSTRAP_INTERVALS.csv", intervals)
    write_json(RESULTS / "MLFLOW_RUNS.json", mlflow_runs)
    write_json(RESULTS / "FINAL_STATUS.json", status)
    render_reports(rows, aggregates, status)
    print(json.dumps(status, indent=2))
    return 0 if supported else 1


if __name__ == "__main__":
    raise SystemExit(main())
