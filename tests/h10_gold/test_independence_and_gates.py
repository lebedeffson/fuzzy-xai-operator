from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.h10_gold.common import ARTIFACT_ROOT, ROOT
from experiments.h10_gold.run_methods import run_split
from experiments.h10_gold.validate_adjudication import validate


FORBIDDEN_BASELINE_NAMES = {"H10Auditor", "SourceLocalizer", "DiagnosticCutSolver", "RepairSetPlanner", "TypedRouteGuard"}


def test_baselines_do_not_import_full_h10_components() -> None:
    for path in (ROOT / "baselines" / "h10_gold").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not FORBIDDEN_BASELINE_NAMES.intersection(names)
        assert not any("audit_h10" in item for item in imports)


def test_oracle_imports_without_framework_package_on_pythonpath() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys;sys.path.insert(0,{str(ROOT)!r});import gold_oracle;print('PASS')"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.stdout.strip() == "PASS"


def test_public_method_inputs_exclude_gold_and_transactions() -> None:
    path = ARTIFACT_ROOT / "data" / "development_inputs.jsonl"
    if not path.exists():
        pytest.skip("generated benchmark inputs are not present")
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    encoded = json.dumps(first)
    assert "source_truth" not in encoded
    assert "repair_truth" not in encoded
    assert "transactions" not in encoded
    assert "mutation_log" not in encoded


def test_sealed_split_fails_without_protocol_lock(tmp_path: Path) -> None:
    lock = ARTIFACT_ROOT / "lock" / "protocol_lock.json"
    if lock.exists():
        pytest.skip("repository contains a valid protocol lock")
    with pytest.raises(RuntimeError, match="protocol lock"):
        run_split("sealed_test", tmp_path)


def test_manual_adjudication_is_not_generated() -> None:
    assert not (ARTIFACT_ROOT / "adjudication" / "reviewer_1.csv").exists()
    assert not (ARTIFACT_ROOT / "adjudication" / "reviewer_2.csv").exists()


def test_adjudication_gate_fails_without_real_reviewers() -> None:
    with pytest.raises(RuntimeError, match="missing real reviewer file"):
        validate(ROOT / "config" / "h10_final_gold_protocol.yaml")
