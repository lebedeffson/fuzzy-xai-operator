from __future__ import annotations

from dataclasses import replace

import pytest

from fuzzyxai.diagnostics import (
    DiagnosticCut,
    MinimalDiagnosticCutFinder,
    RepairCostModel,
    ValidationObligation,
    ValidationResult,
    verify_cut,
)


def _validation(*obligations: ValidationObligation) -> ValidationResult:
    return ValidationResult(
        status="invalid",
        issues=(),
        obligations=obligations,
        checked_contracts=tuple(item.obligation_id for item in obligations),
        passed_contracts=(),
        graph_trace_sha256="abc",
    )


def test_exact_solver_lists_equivalent_optimal_cuts(valid_graph) -> None:
    validation = _validation(
        ValidationObligation("o1", "i1", ("a", "b"), True),
        ValidationObligation("o2", "i2", ("a", "c"), True),
    )
    costs = RepairCostModel({"a": 2.0, "b": 1.0, "c": 1.0})
    cut = MinimalDiagnosticCutFinder().find(valid_graph, validation, costs)
    assert cut.optimal
    assert cut.total_cost == 2.0
    assert set(cut.equivalent_optimal_cuts) == {("a",), ("b", "c")}
    assert cut.defect_atoms == ("a",)


def test_shared_source_covers_multiple_contracts(invalid_route: dict) -> None:
    from fuzzyxai import FuzzyXAI

    report = FuzzyXAI().diagnose(route=invalid_route)
    assert report.minimal_cut is not None
    assert len(report.minimal_cut.defect_atoms) == 1
    assert report.minimal_cut.defect_atoms[0] == "node:preprocessor/violation:source_component"
    assert len(report.minimal_cut.covered_obligations) == 2


def test_large_problem_is_approximate_and_never_claims_optimal(valid_graph) -> None:
    obligations = tuple(
        ValidationObligation(f"o{index}", f"i{index}", (f"a{index}",), True)
        for index in range(30)
    )
    cut = MinimalDiagnosticCutFinder(exact_atom_limit=4).find(valid_graph, _validation(*obligations))
    assert not cut.optimal
    assert cut.solver == "greedy_weighted_hitting_set"
    assert cut.uncovered_obligations == ()


def test_verify_cut_rejects_bad_cost(valid_graph) -> None:
    obligation = ValidationObligation("o1", "i1", ("a",), True)
    cut = MinimalDiagnosticCutFinder().find(valid_graph, _validation(obligation))
    with pytest.raises(ValueError, match="cost"):
        verify_cut(replace(cut, total_cost=99.0), (obligation,))


def test_verify_cut_rejects_unreported_uncovered() -> None:
    cut = DiagnosticCut((), (), 0.0, False, "none", (), (), (), 0.0)
    obligation = ValidationObligation("o1", "i1", (), False)
    with pytest.raises(ValueError, match="unreported"):
        verify_cut(cut, (obligation,))
