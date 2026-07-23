from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from fuzzyxai.diagnostics import (
    Contract,
    DiagnosticValidator,
    RepairExecutionContext,
    RepairExecutor,
    RouteRecertifier,
)
from h10_c3_r4.generator import build_cases
from h10_c3_r4.methods import BASELINES, run_baseline, run_full_h10
from h10_c3_r4.power import (
    _normal_quantile_holm_two_sided,
    simulate_design,
)
from h10_c3_r4.runner import generate_sealed
from h10_c3_r4.runtime import _cut_from_result, execute_and_recertify
from h10_c3_r4.templates import (
    PIPELINE_SCHEMAS,
    audit_banks,
    build_template_bank,
    canonicalize_template,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _template(split: str = "development", stratum: str = "S3"):
    return next(
        item
        for item in build_template_bank(split)
        if item.stratum == stratum
    )


def _renamed(template):
    rename = {
        node.role: f"renamed-{index}"
        for index, node in enumerate(template.node_schema)
    }
    return replace(
        template,
        node_schema=tuple(
            replace(node, role=rename[node.role])
            for node in template.node_schema
        ),
        edge_schema=tuple(
            replace(
                edge,
                source_role=rename[edge.source_role],
                target_role=rename[edge.target_role],
            )
            for edge in template.edge_schema
        ),
        contract_schema=tuple(
            replace(
                contract,
                subject_role=(
                    "edge::"
                    + "::".join(
                        rename[value]
                        for value in contract.subject_role.split("::")[1:]
                    )
                    if contract.subject_role.startswith("edge::")
                    else rename[contract.subject_role]
                ),
                source_roles=tuple(
                    rename[role] for role in contract.source_roles
                ),
            )
            for contract in template.contract_schema
        ),
        candidates=tuple(
            replace(
                candidate,
                source_role=rename[candidate.source_role],
                dependencies=tuple(
                    rename.get(value, value)
                    for value in candidate.dependencies
                ),
            )
            for candidate in template.candidates
        ),
        graph_hash="",
        coverage_hash="",
        mutation_hash="",
        repair_dependency_hash="",
        cost_hash="",
        canonical_hash="",
    )


def test_three_template_banks_have_zero_structural_overlap() -> None:
    report = audit_banks(
        {
            split: build_template_bank(split)
            for split in (
                "development",
                "protocol_validation",
                "sealed",
            )
        }
    )
    assert report["status"] == "PASS"
    assert all(not value for value in report["intersections"].values())


def test_template_overlap_is_detected() -> None:
    development = build_template_bank("development")
    report = audit_banks(
        {
            "development": development,
            "protocol_validation": development,
            "sealed": build_template_bank("sealed"),
        }
    )
    assert report["status"] == "FAIL"
    assert report["intersections"][
        "development:protocol_validation:canonical_hashes"
    ]


def test_node_renaming_does_not_change_canonical_hash() -> None:
    template = _template()
    renamed = canonicalize_template(_renamed(template))
    assert renamed.canonical_hash == template.canonical_hash


def test_structural_change_changes_canonical_hash() -> None:
    template = _template()
    changed = replace(
        template,
        edge_schema=template.edge_schema[:-1],
        graph_hash="",
        coverage_hash="",
        mutation_hash="",
        repair_dependency_hash="",
        cost_hash="",
        canonical_hash="",
    )
    assert canonicalize_template(changed).canonical_hash != template.canonical_hash


def test_pipeline_families_are_structurally_distinct() -> None:
    bank = build_template_bank("development")
    first_hashes = {
        pipeline: next(
            item.graph_hash
            for item in bank
            if item.pipeline_family == pipeline
        )
        for pipeline in PIPELINE_SCHEMAS
    }
    assert len(set(first_hashes.values())) == len(PIPELINE_SCHEMAS)


def test_cases_start_valid_then_become_invalid() -> None:
    case = build_cases((_template(),))[0]
    validator = DiagnosticValidator()
    assert validator.validate(case.valid_graph).valid
    assert not validator.validate(case.mutated_graph).valid


def test_real_executor_changes_graph_and_recertifies() -> None:
    case = build_cases((_template(),))[0]
    result = run_full_h10(case.mutated_graph)
    report = execute_and_recertify(case.mutated_graph, result)
    assert report["graph_changed"]
    assert report["full_recertification_success"]
    assert report["remaining_critical_issue_count"] == 0
    assert report["new_critical_violation_count"] == 0


def test_failed_step_rolls_back_graph_checksum() -> None:
    case = build_cases((_template(),))[0]
    result = run_full_h10(case.mutated_graph)
    cut, validation = _cut_from_result(case.mutated_graph, result)
    from fuzzyxai.diagnostics import ActionableRepairPlanner

    plan = ActionableRepairPlanner().plan(
        case.mutated_graph,
        validation.issues,
        cut,
    )

    def ineffective(graph, step):
        del step
        return replace(graph, metadata={**graph.metadata, "attempt": True})

    context = RepairExecutionContext(
        handlers={
            step.operation: ineffective for step in plan.steps
        },
        approved_step_ids=frozenset(
            step.step_id for step in plan.steps
        ),
        allow_external_changes=True,
        satisfied_preconditions=frozenset(
            condition
            for step in plan.steps
            for condition in step.preconditions
        ),
    )
    after, execution = RepairExecutor().execute(
        case.mutated_graph,
        plan,
        context,
    )
    assert after.trace_sha256 == case.mutated_graph.trace_sha256
    failed_verification = [
        result
        for result in execution
        if result.status == "verification_failed"
    ]
    assert failed_verification
    assert all(result.rollback_verified for result in failed_verification)


def test_new_critical_issue_is_worsened() -> None:
    case = build_cases((_template(),))[0]
    result = run_full_h10(case.mutated_graph)
    cut, validation = _cut_from_result(case.mutated_graph, result)
    from fuzzyxai.diagnostics import (
        ActionableRepairPlanner,
        StepExecutionResult,
    )

    plan = ActionableRepairPlanner().plan(
        case.mutated_graph,
        validation.issues,
        cut,
    )
    after = replace(
        case.mutated_graph,
        contracts=(
            *case.mutated_graph.contracts,
            Contract(
                contract_id="new-critical",
                kind="equals",
                subject_id=case.mutated_graph.nodes[0].node_id,
                field="new_field",
                expected="required",
                severity="error",
            ),
        ),
    )
    execution = tuple(
        StepExecutionResult(step.step_id, "completed", True)
        for step in plan.steps
    )
    recertified = RouteRecertifier().recertify(
        case.mutated_graph,
        after,
        plan,
        execution,
    )
    assert recertified.status == "worsened"
    assert recertified.new_critical_issues


def test_obligation_coverage_without_recertification_is_not_success() -> None:
    case = build_cases((_template(),))[0]
    baseline = run_baseline("simple_or", case.mutated_graph)
    cut, validation = _cut_from_result(case.mutated_graph, baseline)
    assert not cut.uncovered_obligations
    from fuzzyxai.diagnostics import RepairPlan

    empty_plan = RepairPlan(
        "coverage-only",
        cut,
        (),
        0.0,
        True,
        (),
        "coverage-is-not-recertification",
    )
    report = RouteRecertifier().recertify(
        case.mutated_graph,
        case.mutated_graph,
        empty_plan,
        (),
    )
    assert report.status != "full_success"


def test_gold_and_baselines_have_forbidden_import_independence() -> None:
    gold_source = (
        REPO_ROOT / "gold_oracle" / "h10_c3_r4_oracle.py"
    ).read_text(encoding="utf-8")
    method_source = (
        REPO_ROOT
        / "experiments"
        / "h10_c3_r4"
        / "src"
        / "h10_c3_r4"
        / "methods.py"
    ).read_text(encoding="utf-8")
    ast.parse(gold_source)
    ast.parse(method_source)
    forbidden_gold = (
        "fuzzyxai",
        "MinimalDiagnosticCutFinder",
        "ActionableRepairPlanner",
    )
    forbidden_baseline = (
        "gold_oracle",
        "mutation_log",
        "reverse_candidate",
    )
    assert not any(token in gold_source for token in forbidden_gold)
    assert not any(token in method_source for token in forbidden_baseline)
    assert len(BASELINES) == 7


def _power_rows(template_count: int = 80) -> list[dict[str, object]]:
    rows = []
    for pipeline in range(6):
        for template in range(template_count):
            baseline_value = int((template + pipeline) % 10 < 8)
            for method, value in (
                ("weighted_greedy", baseline_value),
                ("full_h10", 1),
            ):
                rows.append(
                    {
                        "case_id": f"p{pipeline}:t{template}",
                        "template_hash": f"p{pipeline}:t{template}",
                        "pipeline_family": f"pipeline-{pipeline}",
                        "stratum": ("S2", "S3", "S4", "S5")[
                            template % 4
                        ],
                        "method": method,
                        "optimal_set_membership": value,
                    }
                )
    return rows


def test_power_increases_with_independent_templates() -> None:
    rows = _power_rows()
    small = simulate_design(
        rows,
        claim="H10-C3a",
        baseline="weighted_greedy",
        metric="optimal_set_membership",
        pipeline_families=6,
        templates_per_family=5,
        cases_per_template=1,
        simulations=1000,
        seed=901,
        margin=0.04,
        intracluster_correlation=0.70,
    )
    large = simulate_design(
        rows,
        claim="H10-C3a",
        baseline="weighted_greedy",
        metric="optimal_set_membership",
        pipeline_families=6,
        templates_per_family=60,
        cases_per_template=1,
        simulations=1000,
        seed=901,
        margin=0.04,
        intracluster_correlation=0.70,
    )
    assert large["point_power"] >= small["point_power"]
    assert (
        large["effective_independent_units"]
        > small["effective_independent_units"]
    )


def test_template_copies_add_little_effective_sample() -> None:
    rows = _power_rows()
    one = simulate_design(
        rows,
        claim="H10-C3a",
        baseline="weighted_greedy",
        metric="optimal_set_membership",
        pipeline_families=6,
        templates_per_family=40,
        cases_per_template=1,
        simulations=100,
        seed=902,
        margin=0.04,
        intracluster_correlation=0.90,
    )
    four = simulate_design(
        rows,
        claim="H10-C3a",
        baseline="weighted_greedy",
        metric="optimal_set_membership",
        pipeline_families=6,
        templates_per_family=40,
        cases_per_template=4,
        simulations=100,
        seed=902,
        margin=0.04,
        intracluster_correlation=0.90,
    )
    assert four["effective_independent_units"] < 1.3 * one[
        "effective_independent_units"
    ]


def test_power_uses_holm_adjusted_critical_value() -> None:
    assert _normal_quantile_holm_two_sided() > 1.96


def test_sealed_generation_is_blocked_before_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import h10_c3_r4.runner as runner

    monkeypatch.setattr(runner, "ARTIFACT_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        generate_sealed()


def test_repeated_sealed_opening_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import h10_c3_r4.runner as runner

    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / "sealed_status.json").write_text(
        '{"sealed_opening_count": 1}\n',
        encoding="utf-8",
    )
    approval = tmp_path / "approval.json"
    approval.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ARTIFACT_ROOT", tmp_path)
    with pytest.raises(PermissionError, match="cannot be repeated"):
        runner.score_sealed(str(approval))
