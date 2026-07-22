from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from baselines.h10_gold import IndependentIfElse, TypedRouteWithoutReasoning
from fuzzyxai.audit_h10.gold_benchmark import FullH10GoldAuditor

from .common import ARTIFACT_ROOT, PRIVATE_ROOT, read_jsonl, write_json
from .metrics import best_set_scores, repair_action_cost, set_scores


METHODS = (IndependentIfElse, TypedRouteWithoutReasoning, FullH10GoldAuditor)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def _score(case: dict[str, Any], truth: dict[str, Any], method: Any) -> dict[str, Any]:
    started = perf_counter()
    result = method.diagnose(case)
    latency_ms = (perf_counter() - started) * 1000.0
    source_precision, source_recall, source_f1, source_jaccard = set_scores(
        truth["source_truth"], result["source_elements"]
    )
    repair_precision, repair_recall, repair_f1, repair_jaccard = set_scores(
        truth["repair_truth"], result["repair_actions"]
    )
    cut_precision, cut_recall, cut_f1, cut_jaccard = best_set_scores(
        truth["optimal_cuts"], result["cut_nodes"]
    )
    cut_exact = any(set(candidate) == set(result["cut_nodes"]) for candidate in truth["optimal_cuts"])
    predicted_cut_cost = sum(float(case["repair_costs"].get(node, 20.0)) for node in set(result["cut_nodes"]))
    optimal_cut_cost = float(truth["optimal_cut_cost"])
    predicted_repair_cost = sum(repair_action_cost(item) for item in set(result["repair_actions"]))
    optimal_repair_cost = sum(repair_action_cost(item) for item in set(truth["repair_truth"]))
    truth_valid = truth["case_type"] == "clean"
    return {
        "case_id": case["case_id"],
        "pipeline_id": case["pipeline_id"],
        "modality": case["modality"],
        "split": case["split"],
        "case_type": truth["case_type"],
        "severity": truth["severity"],
        "unknown": truth["unknown"],
        "insufficient_evidence": truth["insufficient_evidence"],
        "method": result["method"],
        "predicted_status": result["route_status"],
        "source_precision": source_precision,
        "source_recall": source_recall,
        "source_f1": source_f1,
        "source_jaccard": source_jaccard,
        "repair_precision": repair_precision,
        "repair_recall": repair_recall,
        "repair_f1": repair_f1,
        "repair_jaccard": repair_jaccard,
        "cut_precision": cut_precision,
        "cut_recall": cut_recall,
        "cut_f1": cut_f1,
        "cut_jaccard": cut_jaccard,
        "cut_exact": float(cut_exact),
        "cut_cost_regret": max(0.0, predicted_cut_cost - optimal_cut_cost),
        "repair_cost_regret": max(0.0, predicted_repair_cost - optimal_repair_cost),
        "extra_repair_actions": len(set(result["repair_actions"]) - set(truth["repair_truth"])),
        "recertified": bool(result.get("recertified", False)),
        "false_certification": float(not truth_valid and result["route_status"] == "valid"),
        "false_block": float(truth_valid and result["route_status"] != "valid"),
        "abstained": bool(result.get("abstained", False)),
        "false_concrete_unknown": float(truth["unknown"] and not result.get("abstained", False)),
        "runtime_ms": latency_ms,
        "trace": result.get("trace", ""),
    }


def run_split(split: str, output_dir: Path) -> Path:
    if split == "sealed_test" and not (ARTIFACT_ROOT / "lock" / "protocol_lock.json").exists():
        raise RuntimeError("sealed scoring forbidden before adjudication and protocol lock")
    inputs = read_jsonl(ARTIFACT_ROOT / "data" / f"{split}_inputs.jsonl")
    truths = read_jsonl(PRIVATE_ROOT / f"{split}_truth.jsonl")
    truth_by_id = {row["case_id"]: row for row in truths}
    if set(truth_by_id) != {row["case_id"] for row in inputs}:
        raise RuntimeError(f"input/truth identity mismatch for {split}")
    rows: list[dict[str, Any]] = []
    methods = [factory() for factory in METHODS]
    for case in inputs:
        if "transactions" in json.dumps(case) or "source_truth" in case or "repair_truth" in case:
            raise RuntimeError("method input contains Gold or mutation log")
        truth = truth_by_id[case["case_id"]]
        rows.extend(_score(case, truth, method) for method in methods)
    output = output_dir / f"{split}_raw_results.csv"
    _write_csv(output, rows)
    summary = []
    for method_name in sorted({row["method"] for row in rows}):
        group = [row for row in rows if row["method"] == method_name]
        for population in ("all", "single", "composite", "unknown_ambiguous"):
            selected = group if population == "all" else [row for row in group if row["case_type"] == population]
            if not selected:
                continue
            summary.append(
                {
                    "method": method_name,
                    "population": population,
                    "cases": len(selected),
                    "source_f1": sum(float(row["source_f1"]) for row in selected) / len(selected),
                    "repair_f1": sum(float(row["repair_f1"]) for row in selected) / len(selected),
                    "false_certification": sum(float(row["false_certification"]) for row in selected) / len(selected),
                    "false_block": sum(float(row["false_block"]) for row in selected) / len(selected),
                    "cut_exact": sum(float(row["cut_exact"]) for row in selected) / len(selected),
                    "cut_cost_regret": sum(float(row["cut_cost_regret"]) for row in selected) / len(selected),
                    "repair_cost_regret": sum(float(row["repair_cost_regret"]) for row in selected) / len(selected),
                    "runtime_ms": sum(float(row["runtime_ms"]) for row in selected) / len(selected),
                }
            )
    _write_csv(output_dir / f"{split}_summary.csv", summary)
    write_json(output_dir / f"{split}_run.json", {"split": split, "cases": len(inputs), "methods": [item.name for item in methods]})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "protocol_validation", "sealed_test"), required=True)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_ROOT / "exploratory")
    args = parser.parse_args()
    if args.split == "sealed_test":
        lock_path = ARTIFACT_ROOT / "lock" / "protocol_lock.json"
        if not lock_path.exists():
            raise RuntimeError("sealed scoring forbidden before adjudication and protocol lock")
    run_split(args.split, args.output_dir)


if __name__ == "__main__":
    main()
