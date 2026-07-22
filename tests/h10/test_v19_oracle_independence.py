from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

from experiments.h10.oracle_v19 import MutationOperation, build_truth, independent_optimal_cuts
from experiments.h10.routes import build_route
from experiments.h10.audit_methodology import audit


def test_oracle_does_not_import_evaluated_h10_components() -> None:
    path = Path("experiments/h10/oracle_v19.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    failures = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(("fuzzyxai.audit_h10", "baselines.h10")):
            failures.append(node.lineno)
        if isinstance(node, ast.Import):
            failures.extend(node.lineno for item in node.names if item.name.startswith(("fuzzyxai.audit_h10", "baselines.h10")))
    assert not failures


def test_oracle_runs_when_full_h10_is_absent_from_pythonpath(tmp_path: Path) -> None:
    oracle = Path("experiments/h10/oracle_v19.py").resolve()
    isolated = tmp_path / "oracle_v19.py"
    isolated.write_bytes(oracle.read_bytes())
    script = (
        "import runpy; "
        f"ns=runpy.run_path({str(isolated)!r}); "
        "cuts,cost=ns['independent_optimal_cuts'](({'a','b'},),{'a':1.0,'b':1.0}); "
        "assert cuts == (('a',),('b',)) and cost == 1.0"
    )
    subprocess.run([sys.executable, "-I", "-c", script], cwd=tmp_path, check=True)


def test_independent_oracle_keeps_all_equal_cost_optimal_cuts() -> None:
    cuts, cost = independent_optimal_cuts((frozenset(("a", "b")),), {"a": 1.0, "b": 1.0})
    assert cuts == (("a",), ("b",))
    assert cost == 1.0


def test_truth_is_derived_from_mutation_log_and_repair_root() -> None:
    route = build_route("d", "tabular", "x")
    operation = MutationOperation("version_mismatch", "artifact_integrity", "moderate", ("model_version", "model_id"), ("model_registry",))
    truth = build_truth(
        case_id="case",
        operations=(operation,),
        dependency_paths=route.dependency_paths,
        repair_costs=route.repair_costs,
        unknown=False,
        insufficient=False,
    )
    assert truth.source_nodes == ("model_registry",)
    assert truth.repair_sets == (("model_registry",),)
    assert truth.optimal_cuts


def test_repository_audit_detects_static_source_truth_coupling() -> None:
    report = audit()
    assert report["status"] == "INVALID_FOR_PRIMARY_COMPARATIVE_CLAIMS"
    assert report["findings"]["source_truth_derived_from_actual_mutated_graph_nodes"] is False
    assert report["findings"]["source_catalog_semantically_independent"] is False
    assert report["affected_claims"] == ["H10-L", "H10-R"]
