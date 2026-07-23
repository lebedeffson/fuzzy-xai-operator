from __future__ import annotations

import pytest

from h10_c2.data import generate_cases
from h10_c2.models import Case, GoldRecord
from h10_c2.oracle import derive_gold, validate_gold
from h10_c2.oracle.equivalent_cuts import enumerate_optimal_cuts


def test_oracle_enumerates_all_equal_cost_cuts() -> None:
    cuts, cost, explored = enumerate_optimal_cuts((("a", "b"), ("a", "b")), {"a": 1.0, "b": 1.0})
    assert cuts == (("a",), ("b",))
    assert cost == 1.0
    assert explored >= 2


def test_oracle_handles_multi_element_cut_and_costs() -> None:
    cuts, cost, _ = enumerate_optimal_cuts((("a", "b"), ("c",)), {"a": 1, "b": 2, "c": 0})
    assert cuts == (("a", "c"),)
    assert cost == 1.0


def test_oracle_handles_empty_route() -> None:
    assert enumerate_optimal_cuts((), {})[:2] == (((),), 0.0)


def test_oracle_rejects_uncovered_obligation() -> None:
    with pytest.raises(ValueError, match="no repair candidate"):
        enumerate_optimal_cuts(((),), {})


def test_oracle_marks_oversized_case_uncertified() -> None:
    case = generate_cases("development", 1, seed=9)[0]
    obligations = tuple(
        {"obligation_id": f"o:{index}", "candidates": [f"a:{index}"], "repairable": True}
        for index in range(19)
    )
    costs = {f"a:{index}": 1.0 for index in range(19)}
    oversized = Case(**{**case.__dict__, "public_obligations": obligations, "repair_costs": costs})
    assert derive_gold(oversized).gold_status == "uncertified"


def test_gold_validator_rejects_bad_cost() -> None:
    case = next(item for item in generate_cases("development", 20, seed=3) if item.transactions)
    gold = derive_gold(case)
    broken = GoldRecord(**{**gold.__dict__, "optimal_cost": float(gold.optimal_cost or 0) + 1})
    with pytest.raises(ValueError, match="cost"):
        validate_gold(case, broken)


def test_method_view_removes_private_truth() -> None:
    case = generate_cases("development", 20, seed=3)[12]
    view = case.method_view()
    assert view.transactions == ()
    assert view.clean_route == {}
    assert view.public_obligations == ()
    assert "transactions" not in case.public_dict()
    assert "clean_route" not in case.public_dict()
    assert "public_obligations" not in case.public_dict()

