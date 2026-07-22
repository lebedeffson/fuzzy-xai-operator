from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import random
from pathlib import Path
from typing import Any

from gold_oracle import (
    apply_transaction,
    derive_broken_paths,
    derive_repair_truth,
    derive_source_truth,
    enumerate_optimal_cuts,
)

from .common import ARTIFACT_ROOT, PRIVATE_ROOT, ROOT, load_config, sha256_file, write_json, write_jsonl
from .pipelines import pipeline_graphs, propagate_contract_symptoms


KNOWN_OPERATIONS = (
    "remove_node",
    "corrupt_checksum",
    "replace_model_binding",
    "replace_node_version",
    "remove_edge",
    "replace_preprocessing",
    "replace_reference_population",
    "replace_explainer_version",
    "replace_feature_schema",
)
HELD_OUT_OPERATIONS = ("add_false_dependency", "reorder_components", "mix_run_artifact")
COMPOSITION_TEMPLATES = (
    ("corrupt_checksum", "replace_model_binding"),
    ("replace_preprocessing", "replace_explainer_version"),
    ("remove_edge", "replace_reference_population"),
    ("replace_node_version", "replace_feature_schema"),
    ("replace_preprocessing", "corrupt_checksum", "replace_reference_population"),
    ("remove_edge", "replace_model_binding", "replace_explainer_version"),
    ("replace_node_version", "replace_reference_population", "corrupt_checksum"),
    ("replace_feature_schema", "remove_edge", "replace_model_binding"),
)


def _replacement(seed: str, severity: str) -> str:
    suffix = {"subtle": "registered-v1.1", "moderate": "stale-v2", "severe": "incompatible-v9"}[severity]
    return f"{seed}:{suffix}"


def operation_parameters(operation: str, graph: dict[str, Any], severity: str, rng: random.Random) -> dict[str, Any]:
    edges = [(item["source"], item["target"]) for item in graph["edges"]]
    if operation == "remove_node":
        return {"node_id": rng.choice(("canonical", "calibration")), "severity": severity}
    if operation == "corrupt_checksum":
        return {"node_id": "canonical", "field": "checksum", "value": _replacement("sha256", severity), "severity": severity}
    if operation == "replace_model_binding":
        return {"node_id": "canonical", "field": "model_binding", "value": _replacement("other-model", severity), "severity": severity}
    if operation == "replace_node_version":
        return {"node_id": "calibration", "field": "version", "value": _replacement("calibration", severity), "severity": severity}
    if operation == "remove_edge":
        source, target = rng.choice(edges)
        return {"source": source, "target": target, "severity": severity}
    if operation == "add_false_dependency":
        return {"source": "calibration", "target": "explainer", "severity": severity}
    if operation == "reorder_components":
        return {"first": "preprocessor", "second": "model", "third": "explainer", "severity": severity}
    if operation == "mix_run_artifact":
        return {"node_id": "canonical", "field": "run_id", "value": _replacement("foreign-run", severity), "severity": severity}
    if operation == "replace_preprocessing":
        return {"node_id": "preprocessor", "field": "version", "value": _replacement("preprocessor", severity), "severity": severity}
    if operation == "replace_reference_population":
        return {"node_id": "reference", "field": "reference_population", "value": _replacement("near-domain", severity), "severity": severity}
    if operation == "replace_explainer_version":
        return {"node_id": "explainer", "field": "version", "value": _replacement("explainer", severity), "severity": severity}
    if operation == "replace_feature_schema":
        return {"node_id": "preprocessor", "field": "schema", "value": _replacement("schema", severity), "severity": severity}
    raise ValueError(operation)


def _repair_costs(clean: dict[str, Any]) -> dict[str, float]:
    costs = {f"node:{node['id']}": float(node["repair_cost"]) for node in clean["nodes"]}
    for edge in clean["edges"]:
        costs[f"edge:{edge['source']}->{edge['target']}"] = 1.0
    return costs


def _case_counts(size: int) -> list[str]:
    counts = {
        "clean": int(size * 0.20),
        "single": int(size * 0.20),
        "composite": int(size * 0.40),
    }
    counts["unknown_ambiguous"] = size - sum(counts.values())
    return [category for category, count in counts.items() for _ in range(count)]


def _split_categories(config: dict[str, Any], rng: random.Random) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for split, size in config["splits"].items():
        categories = _case_counts(int(size))
        rng.shuffle(categories)
        rows.extend((split, category) for category in categories)
    return rows


def _operation_plan(category: str, rng: random.Random) -> tuple[str, ...]:
    if category == "clean":
        return ()
    if category == "single":
        return (rng.choice(KNOWN_OPERATIONS),)
    if category == "composite":
        return rng.choice(COMPOSITION_TEMPLATES)
    if rng.random() < 0.5:
        return (rng.choice(HELD_OUT_OPERATIONS),)
    composable_known = tuple(item for item in KNOWN_OPERATIONS if item != "remove_node")
    return (rng.choice(composable_known), rng.choice(HELD_OUT_OPERATIONS))


def _case(
    clean_template: dict[str, Any],
    *,
    pipeline_id: str,
    split: str,
    category: str,
    index: int,
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    clean = copy.deepcopy(clean_template)
    graph = copy.deepcopy(clean)
    operations = _operation_plan(category, rng)
    severity = ("subtle", "moderate", "severe")[index % 3] if operations else "none"
    transactions = []
    for op_index, operation in enumerate(operations):
        parameters = operation_parameters(operation, graph, severity, rng)
        graph, transaction = apply_transaction(
            graph,
            transaction_id=f"{pipeline_id}-{split}-{index:04d}-tx{op_index + 1}",
            operation=operation,
            parameters=parameters,
        )
        transactions.append(transaction)
    low_level_mutated = copy.deepcopy(graph)
    observed = propagate_contract_symptoms(clean, graph)
    case_id = f"{pipeline_id}-{split}-{index:04d}"
    broken_paths = derive_broken_paths(clean, observed)
    costs = _repair_costs(clean)
    cut = enumerate_optimal_cuts(broken_paths, costs)
    unknown = category == "unknown_ambiguous"
    insufficient = unknown and index % 2 == 0
    if insufficient:
        # Mask one directly relevant node to represent genuinely unavailable evidence.
        direct = derive_source_truth(tuple(transactions))
        direct_nodes = {item.split(":", 1)[1] for item in direct if item.startswith("node:")}
        for node in observed["nodes"]:
            if node["id"] in direct_nodes:
                node["checksum"] = None
                node["version"] = None
                break
    method_input = {
        "case_id": case_id,
        "pipeline_id": pipeline_id,
        "modality": clean["modality"],
        "split": split,
        "registered_graph": clean,
        "observed_graph": observed,
        "repair_costs": costs,
    }
    truth = {
        "case_id": case_id,
        "pipeline_id": pipeline_id,
        "split": split,
        "case_type": category,
        "severity": severity,
        "unknown": unknown,
        "insufficient_evidence": insufficient,
        "source_truth": list(derive_source_truth(tuple(transactions))),
        "repair_truth": list(derive_repair_truth(tuple(transactions))),
        "broken_paths": [list(path) for path in broken_paths],
        "optimal_cuts": [list(item) for item in cut.optimal_cuts],
        "optimal_cut_cost": cut.optimal_cost,
        "transactions": [item.as_dict() for item in transactions],
    }
    blind = {
        "case_id": case_id,
        "pipeline_id": pipeline_id,
        "case_type": category,
        "clean_graph": clean,
        "mutated_graph": low_level_mutated,
        "mutation_transactions": [
            {
                "transaction_id": item.transaction_id,
                "operation": item.operation,
                "parameters": item.parameters,
                "inverse_operation": item.inverse_operation,
            }
            for item in transactions
        ],
    }
    return method_input, truth, blind


def _adjudication_sample(blind_rows: list[dict[str, Any]], config: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    required = {"single": 50, "composite": 100, "unknown_ambiguous": 50}
    selected: list[dict[str, Any]] = []
    for category, count in required.items():
        candidates = [row for row in blind_rows if row["case_type"] == category]
        selected.extend(rng.sample(candidates, count))
    rng.shuffle(selected)
    assert len(selected) == int(config["adjudication"]["cases"])
    return selected


def _write_adjudication_template(path: Path, case_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "source_elements_json", "optimal_cuts_json", "repair_actions_json", "ambiguous", "notes"])
        for case_id in case_ids:
            writer.writerow([case_id, "", "", "", "", ""])


def generate(config_path: Path) -> None:
    config = load_config(config_path)
    seed = int(config["seed"])
    all_inputs: dict[str, list[dict[str, Any]]] = {name: [] for name in config["splits"]}
    all_truth: dict[str, list[dict[str, Any]]] = {name: [] for name in config["splits"]}
    protocol_blind: list[dict[str, Any]] = []
    graphs = pipeline_graphs()
    for pipeline_offset, pipeline_id in enumerate(config["pipelines"]):
        rng = random.Random(seed + pipeline_offset * 100_003)
        assignments = _split_categories(config, rng)
        for index, (split, category) in enumerate(assignments):
            method_input, truth, blind = _case(
                graphs[pipeline_id],
                pipeline_id=pipeline_id,
                split=split,
                category=category,
                index=index,
                rng=rng,
            )
            all_inputs[split].append(method_input)
            all_truth[split].append(truth)
            if split == "protocol_validation":
                protocol_blind.append(blind)
    data_dir = ARTIFACT_ROOT / "data"
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for split in config["splits"]:
        input_path = data_dir / f"{split}_inputs.jsonl"
        truth_path = PRIVATE_ROOT / f"{split}_truth.jsonl"
        write_jsonl(input_path, all_inputs[split])
        write_jsonl(truth_path, all_truth[split])
        manifest_rows.append(
            {
                "split": split,
                "cases": len(all_inputs[split]),
                "input_sha256": sha256_file(input_path),
                "truth_sha256": sha256_file(truth_path),
                "truth_location": "private_untracked",
            }
        )
    adjudication_rng = random.Random(seed + 9_999_991)
    blind_sample = _adjudication_sample(protocol_blind, config, adjudication_rng)
    adjudication_path = ARTIFACT_ROOT / "adjudication" / "blind_cases.jsonl"
    write_jsonl(adjudication_path, blind_sample)
    for reviewer in (1, 2):
        _write_adjudication_template(
            ARTIFACT_ROOT / "adjudication" / f"reviewer_{reviewer}_template.csv",
            [row["case_id"] for row in blind_sample],
        )
    operation_counts: dict[str, int] = {}
    case_counts: dict[str, int] = {}
    for rows in all_truth.values():
        for row in rows:
            case_counts[row["case_type"]] = case_counts.get(row["case_type"], 0) + 1
            for transaction in row["transactions"]:
                operation_counts[transaction["operation"]] = operation_counts.get(transaction["operation"], 0) + 1
    manifest = {
        "study_id": config["study_id"],
        "phase": "generated_prelock",
        "generator": "transaction_log_and_graph_diff",
        "pipeline_count": len(graphs),
        "case_count": sum(len(rows) for rows in all_inputs.values()),
        "case_counts": case_counts,
        "operation_counts": operation_counts,
        "splits": manifest_rows,
        "adjudication_cases": len(blind_sample),
        "adjudication_sha256": sha256_file(adjudication_path),
        "gold_uses_fault_taxonomy": False,
        "sealed_opened": False,
    }
    write_json(ARTIFACT_ROOT / "h10_final_gold_manifest.json", manifest)
    protocol_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    write_json(
        ARTIFACT_ROOT / "lock" / "prelock_manifest.json",
        {"protocol_path": str(config_path.relative_to(ROOT)), "protocol_sha256": protocol_hash, "sealed_opened": False},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "h10_final_gold_protocol.yaml")
    args = parser.parse_args()
    generate(args.config)


if __name__ == "__main__":
    main()
