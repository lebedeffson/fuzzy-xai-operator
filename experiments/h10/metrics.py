from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Any, Callable, Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from baselines.h10.common import BaselineResult
from fuzzyxai.audit_h10.models import AuditDiagnosis

from .mutations import MutationTruth


def set_scores(expected: Iterable[str], predicted: Iterable[str]) -> tuple[float, float, float, float]:
    left, right = set(expected), set(predicted)
    if not left and not right:
        return 1.0, 1.0, 1.0, 1.0
    precision = len(left & right) / len(right) if right else 0.0
    recall = len(left & right) / len(left) if left else float(not right)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = len(left | right)
    return precision, recall, f1, len(left & right) / union if union else 1.0


def best_alternative_scores(
    alternatives: tuple[tuple[str, ...], ...], predicted: tuple[str, ...]
) -> tuple[float, float, float, float, tuple[str, ...]]:
    candidates = alternatives or ((),)
    scored = [(*set_scores(item, predicted), item) for item in candidates]
    return max(scored, key=lambda row: (row[2], row[3], row[1], row[0], tuple(row[4])))


def normalize(value: AuditDiagnosis | BaselineResult) -> dict[str, Any]:
    if isinstance(value, AuditDiagnosis):
        return {
            "route_status": value.route_status,
            "parent_family": value.fault.parent_family,
            "leaf_type": value.fault.leaf_type,
            "source_nodes": value.source_nodes,
            "cut_nodes": value.diagnostic_cut.cut_nodes,
            "repair_nodes": tuple(item.target for item in value.repair_set),
            "repair_fields": tuple(dict.fromkeys(field for item in value.repair_set for field in item.affected_fields)),
            "unknown": value.fault.unknown,
            "abstained": value.fault.abstained_at_leaf,
            "confidence": value.fault.leaf_confidence,
            "anomaly_score": float(value.details.get("anomaly_score", 0.0)),
            "recertified": value.recertified,
            "trace": value.trace,
            "cut_cost": value.diagnostic_cut.total_cost,
            "cut_optimal": value.diagnostic_cut.optimal,
            "cut_runtime_ms": value.diagnostic_cut.runtime_ms,
        }
    row = asdict(value)
    row.update({"recertified": None, "trace": b"", "cut_cost": 0.0, "cut_optimal": False, "cut_runtime_ms": 0.0, "repair_fields": tuple(row.get("repair_nodes", ()))})
    return row


def evaluate_method(
    name: str,
    diagnose: Callable[[Any], AuditDiagnosis | BaselineResult],
    cases: list[tuple[Any, MutationTruth]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route, truth in cases:
        started = perf_counter()
        result = normalize(diagnose(route))
        latency_ms = (perf_counter() - started) * 1000.0
        source_precision, source_recall, source_f1, _ = set_scores(truth.source_nodes, tuple(result["source_nodes"]))
        repair_precision, repair_recall, repair_f1, repair_jaccard, best_repair = best_alternative_scores(
            truth.repair_sets, tuple(result["repair_nodes"])
        )
        _, _, cut_f1, cut_jaccard, best_cut = best_alternative_scores(truth.optimal_cuts, tuple(result["cut_nodes"]))
        cut_cost = float(result["cut_cost"] or sum(route.repair_costs.get(node, 1.0) for node in result["cut_nodes"]))
        expected_parent = truth.parent_families[0] if len(truth.parent_families) == 1 else None
        expected_leaf = truth.leaf_types[0] if len(truth.leaf_types) == 1 else None
        rows.append(
            {
                "case_id": truth.case_id,
                "dataset": route.dataset_id,
                "modality": route.modality,
                "method": name,
                "truth_status": truth.route_status,
                "truth_parent": expected_parent,
                "truth_leaf": expected_leaf,
                "truth_unknown": truth.unknown,
                "severity": truth.severity,
                "composite": truth.composite,
                "predicted_status": result["route_status"],
                "predicted_parent": result["parent_family"],
                "predicted_leaf": result["leaf_type"],
                "predicted_unknown": result["unknown"],
                "abstained": result["abstained"],
                "known_confidence": result["confidence"],
                "anomaly_score": result["anomaly_score"],
                "source_precision": source_precision,
                "source_recall": source_recall,
                "source_f1": source_f1,
                "repair_precision": repair_precision,
                "repair_recall": repair_recall,
                "repair_f1": repair_f1,
                "repair_jaccard": repair_jaccard,
                "cut_exact": float(any(set(item) == set(result["cut_nodes"]) for item in truth.optimal_cuts)),
                "cut_f1": cut_f1,
                "cut_jaccard": cut_jaccard,
                "cut_cost_ratio": cut_cost / truth.optimal_cost if truth.optimal_cost else 1.0,
                "cut_extra_nodes": len(set(result["cut_nodes"]) - set(best_cut)),
                "repair_extra_nodes": len(set(result["repair_nodes"]) - set(best_repair)),
                "cut_runtime_ms": result["cut_runtime_ms"],
                "cut_optimal": result["cut_optimal"],
                "recertified": result["recertified"],
                "false_certification": float(truth.route_status != "valid" and result["route_status"] == "valid"),
                "false_block": float(truth.route_status == "valid" and result["route_status"] != "valid"),
                "parent_correct": float(result["parent_family"] == expected_parent),
                "leaf_correct": float(result["leaf_type"] == expected_leaf),
                "unknown_correct": float(bool(result["unknown"]) == truth.unknown),
                "diagnostic_latency_ms": latency_ms,
                "trace_size": len(result["trace"]),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    invalid = [row for row in rows if row["truth_status"] != "valid"]
    known = [row for row in invalid if not row["truth_unknown"]]
    target = known or invalid or rows
    summary = {
        "source_f1": float(np.mean([float(row["source_f1"]) for row in target])),
        "repair_f1": float(np.mean([float(row["repair_f1"]) for row in target])),
        "false_certification": float(np.mean([float(row["false_certification"]) for row in rows])),
        "false_block": float(np.mean([float(row["false_block"]) for row in rows])),
        "parent_correct": float(np.mean([float(row["parent_correct"]) for row in known])) if known else 0.0,
        "leaf_correct": float(np.mean([float(row["leaf_correct"]) for row in known])) if known else 0.0,
        "unknown_correct": float(np.mean([float(row["unknown_correct"]) for row in invalid])) if invalid else 1.0,
        "cut_exact": float(np.mean([float(row["cut_exact"]) for row in invalid])) if invalid else 1.0,
        "cut_jaccard": float(np.mean([float(row["cut_jaccard"]) for row in invalid])) if invalid else 1.0,
        "cut_cost_ratio": float(np.mean([float(row["cut_cost_ratio"]) for row in invalid])) if invalid else 1.0,
        "diagnostic_latency_ms": float(np.mean([float(row["diagnostic_latency_ms"]) for row in rows])),
        "trace_size": float(np.mean([float(row["trace_size"]) for row in rows])),
    }
    y_true = np.asarray([float(row["truth_unknown"]) for row in invalid])
    anomaly = np.asarray([float(row["anomaly_score"]) for row in invalid])
    if len(np.unique(y_true)) == 2:
        summary["unknown_auroc"] = float(roc_auc_score(y_true, anomaly))
        summary["unknown_auprc"] = float(average_precision_score(y_true, anomaly))
    else:
        summary["unknown_auroc"] = float("nan")
        summary["unknown_auprc"] = float("nan")
    return summary
