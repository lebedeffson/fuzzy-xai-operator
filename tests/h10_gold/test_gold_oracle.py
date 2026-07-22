from __future__ import annotations

import ast
import copy
from pathlib import Path

from gold_oracle import apply_transaction, derive_repair_truth, derive_source_truth, enumerate_optimal_cuts
from gold_oracle.mutation_transaction import MutationTransaction
from experiments.h10_gold.common import ARTIFACT_ROOT, PRIVATE_ROOT, read_jsonl
from experiments.h10_gold.pipelines import pipeline_graphs


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = {
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


def test_oracle_has_no_forbidden_import_or_symbol_reference() -> None:
    for path in (ROOT / "gold_oracle").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not FORBIDDEN.intersection(names)
        assert not any("fuzzyxai" in item or "audit_h10" in item for item in imports)


def test_source_and_repair_truth_follow_executed_transaction() -> None:
    graph = pipeline_graphs()["tabular_tree_a"]
    mutated, transaction = apply_transaction(
        graph,
        transaction_id="tx-1",
        operation="replace_preprocessing",
        parameters={"node_id": "preprocessor", "field": "version", "value": "stale-v2"},
    )
    assert graph["nodes"] != mutated["nodes"]
    assert derive_source_truth((transaction,)) == ("node:preprocessor",)
    repairs = derive_repair_truth((transaction,))
    assert len(repairs) == 1
    assert '"operation":"restore_attribute"' in repairs[0]
    assert '"value":"registered-v1"' in repairs[0]


def test_source_truth_does_not_depend_on_operation_name() -> None:
    graph = pipeline_graphs()["tabular_tree_a"]
    _, first = apply_transaction(
        graph,
        transaction_id="tx-a",
        operation="replace_preprocessing",
        parameters={"node_id": "preprocessor", "field": "version", "value": "stale-v2"},
    )
    renamed = copy.deepcopy(first)
    object.__setattr__(renamed, "operation", "opaque_operation_99")
    assert derive_source_truth((first,)) == derive_source_truth((renamed,))


def test_cut_oracle_keeps_all_equal_cost_optima() -> None:
    result = enumerate_optimal_cuts(
        (("node:a", "node:b"), ("node:a", "node:c")),
        {"node:a": 2.0, "node:b": 1.0, "node:c": 1.0},
    )
    assert set(result.optimal_cuts) == {("node:a",), ("node:b", "node:c")}
    assert result.optimal_cost == 2.0


def test_generated_source_truth_matches_executed_deltas() -> None:
    path = PRIVATE_ROOT / "development_truth.jsonl"
    if not path.exists():
        return
    for row in read_jsonl(path)[:100]:
        transactions = tuple(
            MutationTransaction(
                transaction_id=item["transaction_id"],
                operation=item["operation"],
                parameters=item["parameters"],
                changed_nodes=tuple(item["changed_nodes"]),
                changed_edges=tuple(item["changed_edges"]),
                inverse_operation=item["inverse_operation"],
            )
            for item in row["transactions"]
        )
        assert tuple(row["source_truth"]) == derive_source_truth(transactions)
        assert tuple(row["repair_truth"]) == derive_repair_truth(transactions)


def test_generated_manifest_has_registered_size_and_all_operations() -> None:
    path = ARTIFACT_ROOT / "h10_final_gold_manifest.json"
    if not path.exists():
        return
    import json

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["pipeline_count"] == 6
    assert manifest["case_count"] == 4500
    assert manifest["case_counts"] == {"clean": 900, "single": 900, "composite": 1800, "unknown_ambiguous": 900}
    assert len(manifest["operation_counts"]) == 12
    assert all(count > 0 for count in manifest["operation_counts"].values())
