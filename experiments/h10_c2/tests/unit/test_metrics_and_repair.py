from __future__ import annotations

from h10_c2.data import generate_cases
from h10_c2.metrics import score_cut, score_repair
from h10_c2.models import MethodResult
from h10_c2.oracle import derive_gold
from h10_c2.repair import execute_plan


def _fault_case():
    return next(item for item in generate_cases("development", 30, seed=7) if item.transactions)


def test_cut_membership_uses_all_optimal_cuts() -> None:
    case = _fault_case()
    gold = derive_gold(case)
    result = MethodResult(case.case_id, case.pipeline, "test", gold.optimal_cuts[-1], float(gold.optimal_cost or 0))
    score = score_cut(result, gold, case.public_obligations)
    assert score["optimal_cut_set_membership"] is True
    assert score["cut_cost_regret"] == 0.0
    assert score["cut_jaccard_best"] == 1.0


def test_sandbox_recertifies_only_executable_provider_actions() -> None:
    case = _fault_case()
    action = {
        "operation": "restore_from_registered_provider",
        "target": case.transactions[0].target_id,
    }
    result = execute_plan(case.observed_route, case.clean_route, (action,))
    if len(case.transactions) == 1:
        assert result["full_recertification_success"]
    assert result["audit"][0]["precondition_passed"]
    assert "before_sha256" in result["audit"][0]


def test_sandbox_rejects_unknown_action_without_mutating_source() -> None:
    case = _fault_case()
    original = case.observed_route["nodes"][0]["observed_attributes"].copy()
    result = execute_plan(case.observed_route, case.clean_route, ({"operation": "copy_expected", "target": "missing"},))
    assert not result["full_recertification_success"]
    assert result["human_actions"] == 1
    assert case.observed_route["nodes"][0]["observed_attributes"] == original
    scored = score_repair(result, 3.0, 1.0)
    assert scored["plan_cost_regret"] == 2.0

