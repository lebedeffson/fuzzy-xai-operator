from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest
from fuzzyxai.experiments.h10_c5 import _public_row, screen_candidates
from fuzzyxai.incidents import IncidentInput, RepositoryImporter
from fuzzyxai.incidents.audit import audit_greedy, audit_route, audit_rule, audit_traceback


def test_gold_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="gold fields"):
        IncidentInput.from_public_mapping(
            {"instance_id": "x", "repo": "r", "problem_statement": "x", "patch": "secret"}
        )


def test_public_projection_removes_gold() -> None:
    public = _public_row(
        {
            "instance_id": "x",
            "repo": "r",
            "problem_statement": "schema mismatch",
            "FAIL_TO_PASS": ["test_x"],
            "patch": "secret",
            "base_commit": "abc",
        }
    )
    assert "patch" not in public
    assert public["base_commit"] == "abc"


def test_importer_builds_observable_route() -> None:
    incident = IncidentInput.from_public_mapping(
        {
            "instance_id": "x",
            "repo": "owner/repo",
            "base_commit": "abc",
            "problem_statement": "schema mismatch in preprocessing",
            "FAIL_TO_PASS": ("test_schema",),
        }
    )
    route = RepositoryImporter().import_incident(incident)
    assert route.components
    assert all(component.evidence_refs for component in route.components)


def test_all_locked_strategies_return_formal_operations() -> None:
    incident = IncidentInput.from_public_mapping(
        {
            "instance_id": "x",
            "repo": "owner/repo",
            "base_commit": "abc",
            "problem_statement": "version mismatch while loading serialized model",
            "FAIL_TO_PASS": ("test_load",),
        }
    )
    route = RepositoryImporter().import_incident(incident)
    strategies = (audit_traceback, audit_rule, audit_greedy, audit_route)
    for strategy in strategies:
        result = strategy(route)
        assert result.operations


def test_screening_meets_locked_quota_when_source_available() -> None:
    source = Path("/tmp/h10c5_sources/swebench_lite_test.parquet")
    if not source.exists():
        pytest.skip("locked external source is not available")
    screened, selected = screen_candidates(pd.read_parquet(source))
    accepted = [item for item in screened if item.source_index in set(selected)]
    assert len(screened) >= 100
    assert 24 <= len(accepted) <= 30
    assert len({item.repository_id for item in accepted}) >= 6
    assert len({item.contract_family for item in accepted}) >= 4


def test_auditor_module_does_not_reference_gold_patch() -> None:
    module = Path("framework/fuzzyxai/fuzzyxai/incidents/audit.py")
    tree = ast.parse(module.read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not {"patch", "gold", "source_truth", "repair_truth"}.intersection(names)
