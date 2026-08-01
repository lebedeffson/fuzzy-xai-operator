from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
from fuzzyxai.repository_diagnostics.auditor import AuditCandidate, AuditResult
from fuzzyxai.repository_diagnostics.evidence_requests import (
    EvidenceRequestPlanner,
)
from fuzzyxai.repository_diagnostics.executed_slice import ExecutedSliceBuilder
from fuzzyxai.repository_diagnostics.graph import RepositoryGraph, RepositoryNode
from fuzzyxai.repository_diagnostics.practical_recovery import (
    PracticalRepairExecutor,
    PracticalRepairPlanner,
    RegisteredRepairProvider,
)
from fuzzyxai.repository_diagnostics.reporting import build_engineering_report
from fuzzyxai.repository_diagnostics.runtime_events import RuntimeEvent


def _candidate(contract: str) -> AuditCandidate:
    return AuditCandidate(
        "source",
        "fixture/repo",
        "src/core.py",
        "load",
        contract,
        10.0,
        0.8,
        ("failure",),
        (),
        1.0,
        0.0,
    )


def test_executed_slice_keeps_runtime_modalities_per_test() -> None:
    test_id = "tests/test_core.py::test_load"
    events = (
        RuntimeEvent(
            "trace",
            test_id,
            "traceback_frame",
            "src/core.py",
            "load",
        ),
        RuntimeEvent(
            "coverage",
            test_id,
            "coverage",
            "src/core.py",
            "load",
        ),
        RuntimeEvent(
            "read",
            test_id,
            "read",
            "src/core.py",
            "load",
            "models/checkpoint.bin",
        ),
        RuntimeEvent(
            "config",
            test_id,
            "config_read",
            "src/core.py",
            "load",
            "config.yaml",
            "loader.format",
        ),
        RuntimeEvent(
            "dependency",
            test_id,
            "dependency",
            "src/core.py",
            "load",
            "pyproject.toml",
            "serializer",
            "2.0.1",
        ),
    )
    result = ExecutedSliceBuilder().build(events)
    assert len(result) == 1
    assert result[0].failing_test == test_id
    assert result[0].traceback_symbols[0].symbol == "load"
    assert result[0].accessed_artifacts[0].path == "models/checkpoint.bin"
    assert result[0].configuration_reads[0].key == "loader.format"
    assert result[0].dependency_versions == (("serializer", "2.0.1"),)


def test_medium_confidence_result_requests_read_only_evidence() -> None:
    test_id = "tests/test_core.py::test_load"
    graph = RepositoryGraph(
        "fixture/repo",
        "buggy",
        (
            RepositoryNode(
                "runtime",
                "runtime_exception",
                "fixture/repo",
                symbol=test_id,
                attributes={"obligation": "failure"},
            ),
        ),
        (),
        (),
        ("failure",),
    )
    result = AuditResult(
        "O_ROUTE",
        "DIAGNOSIS_CANDIDATES",
        (_candidate("SERIALIZATION"),),
        (),
        (),
        (),
        ("failure",),
        0.0,
        (),
    )
    request = EvidenceRequestPlanner().plan(graph, result)[0]
    assert request.command == (
        "python",
        "-m",
        "pytest",
        test_id,
        "-x",
        "-vv",
        "--showlocals",
    )
    assert request.safety_level == "READ_ONLY_TEST_EXECUTION"


def test_repair_planner_requires_approval_for_source_adapter_change() -> None:
    plan = PracticalRepairPlanner().plan(
        _candidate("DATA_CONTRACT"),
        ("tests/test_core.py::test_load",),
    )
    assert plan.automation_level == "HUMAN_APPROVAL"
    assert plan.executable is True
    assert plan.requires_human_approval is True
    assert plan.rollback == "restore sandbox snapshot"


def test_unregistered_contract_is_localization_only() -> None:
    plan = PracticalRepairPlanner().plan(
        _candidate("UNREGISTERED_CONTRACT"),
        ("tests/test_core.py::test_load",),
    )
    assert plan.automation_level == "LOCALIZATION_ONLY"
    assert plan.executable is False


def test_executor_blocks_unapproved_semiautomatic_repair(tmp_path: Path) -> None:
    plan = PracticalRepairPlanner().plan(
        _candidate("DATA_CONTRACT"),
        ("tests/test_core.py::test_load",),
    )
    executor = PracticalRepairExecutor(
        (
            RegisteredRepairProvider(
                plan.operation,
                lambda _root: None,
            ),
        )
    )
    with pytest.raises(PermissionError, match="human approval"):
        executor.execute(
            plan,
            tmp_path,
            approved=False,
            recertify=lambda _root: True,
        )


def test_executor_requires_registered_provider(tmp_path: Path) -> None:
    plan = PracticalRepairPlanner().plan(
        _candidate("DEPENDENCY_VERSION"),
        ("tests/test_core.py::test_load",),
    )
    with pytest.raises(ValueError, match="provider is missing"):
        PracticalRepairExecutor(()).execute(
            plan,
            tmp_path,
            approved=True,
            recertify=lambda _root: True,
        )


def test_executor_runs_fail_to_pass_regression_and_recertification(
    tmp_path: Path,
) -> None:
    (tmp_path / "state.txt").write_text("broken", encoding="utf-8")
    (tmp_path / "check.py").write_text(
        "from pathlib import Path\n"
        "raise SystemExit(0 if Path('state.txt').read_text() == 'fixed' else 1)\n",
        encoding="utf-8",
    )
    plan = PracticalRepairPlanner().plan(
        _candidate("DEPENDENCY_VERSION"),
        ("check.py",),
    )
    command = (sys.executable, "check.py")
    plan = replace(
        plan,
        fail_to_pass_command=command,
        regression_command=command,
    )
    executor = PracticalRepairExecutor(
        (
            RegisteredRepairProvider(
                plan.operation,
                lambda root: (root / "state.txt").write_text(
                    "fixed",
                    encoding="utf-8",
                ),
            ),
        )
    )
    report = executor.execute(
        plan,
        tmp_path,
        approved=True,
        recertify=lambda root: (root / "state.txt").read_text(
            encoding="utf-8",
        )
        == "fixed",
    )
    assert report.statuses == (
        "BUG_REPRODUCED",
        "REPAIR_EXECUTED",
        "FAIL_TO_PASS_PASS",
        "REGRESSION_PASS",
        "RECERTIFICATION_PASS",
    )
    assert report.recertification_success is True
    assert report.before_sha256 != report.after_sha256


def test_engineering_report_starts_with_actionable_information() -> None:
    test_id = "tests/test_core.py::test_load"
    graph = RepositoryGraph(
        "fixture/repo",
        "buggy",
        (
            RepositoryNode(
                "runtime",
                "runtime_exception",
                "fixture/repo",
                symbol=test_id,
                attributes={"obligation": "failure"},
            ),
        ),
        (),
        (),
        ("failure",),
    )
    result = AuditResult(
        "O_ROUTE",
        "DIAGNOSIS_CANDIDATES",
        (_candidate("SERIALIZATION"),),
        (),
        (),
        (),
        ("failure",),
        0.0,
        (),
    )
    requests = EvidenceRequestPlanner().plan(graph, result)
    plan = PracticalRepairPlanner().plan(
        result.candidates[0],
        (test_id,),
    )
    report = build_engineering_report(graph, result, requests, plan)
    assert report.diagnosis.probable_source == "src/core.py::load"
    assert report.diagnosis.contract == "SERIALIZATION"
    assert report.next_step is not None
    assert report.next_step.command[0:3] == ("python", "-m", "pytest")
    assert report.audit_reference == graph.trace_sha256
