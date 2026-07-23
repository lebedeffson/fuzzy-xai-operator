from __future__ import annotations

import ast

import pytest

from h10_c3.baseline_methods import run_baseline
from h10_c3.fuzzy_method import run_fuzzyxai
from h10_c3.generator import generate_cases
from h10_c3.oracle import derive_gold, oracle_a, oracle_b
from h10_c3.runner import EXPERIMENT_ROOT, score_sealed
from h10_c3.scoring import score


def _case(stratum: str):
    return next(
        item
        for item in generate_cases("development", 20, 231004)
        if item.stratum == stratum and item.repairable
    )


def test_composite_share_is_at_least_seventy_percent() -> None:
    cases = generate_cases("development", 100, 231004)
    composite = sum(item.stratum != "S1" for item in cases) / len(cases)
    assert composite >= 0.70


def test_oracles_agree_and_multiple_optima_are_preserved() -> None:
    case = _case("S4")
    assert oracle_a(case) == oracle_b(case)
    gold = derive_gold(case)
    assert gold.status == "CERTIFIED_MULTIPLE_OPTIMA"
    assert len(gold.optimal_cuts) >= 2


def test_gold_is_derived_from_private_transactions_only() -> None:
    case = _case("S3")
    public = case.method_view()
    assert "mutations" not in public
    assert "optimal_cuts" not in public
    assert "optimal_cost" not in public
    assert all(item in {candidate.atom_id for candidate in case.candidates} for item in case.mutations[0].allowed_inverse_ids)


def test_full_method_beats_weighted_greedy_on_registered_composite() -> None:
    case = _case("S3")
    gold = derive_gold(case)
    full = score(case, gold, run_fuzzyxai(case.method_view()))
    baseline = score(case, gold, run_baseline("weighted_greedy", case.method_view()))
    assert full["optimal_set_membership"] is True
    assert baseline["optimal_set_membership"] is False
    assert full["normalized_cost_regret"] < baseline["normalized_cost_regret"]
    assert full["full_recertification_success"] is True
    assert baseline["full_recertification_success"] is False


def test_single_fault_is_a_parity_control() -> None:
    case = _case("S1")
    gold = derive_gold(case)
    full = score(case, gold, run_fuzzyxai(case.method_view()))
    baseline = score(case, gold, run_baseline("weighted_greedy", case.method_view()))
    assert full["optimal_set_membership"] == baseline["optimal_set_membership"]
    assert full["full_recertification_success"] == baseline["full_recertification_success"]


def test_nonrepairable_case_fails_closed() -> None:
    case = next(
        item
        for item in generate_cases("development", 40, 231004)
        if not item.repairable
    )
    gold = derive_gold(case)
    result = run_fuzzyxai(case.method_view())
    assert gold.status == "NON_REPAIRABLE"
    assert result.status == "insufficient_evidence"


def test_baseline_module_has_no_fuzzyxai_import() -> None:
    path = EXPERIMENT_ROOT / "src" / "h10_c3" / "baseline_methods.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(name.startswith("fuzzyxai") for name in imported)


def test_sealed_scoring_is_fail_closed() -> None:
    with pytest.raises(PermissionError, match="no sealed set"):
        score_sealed()

