from copy import deepcopy

from fuzzyxai.core.proof_package import canonicalize_explain_plan, compute_explain_plan_hash
from fuzzyxai.core.scenario_engine import DEFAULT_HYBRID_PLAN


def _plan() -> dict:
    return deepcopy(DEFAULT_HYBRID_PLAN.__dict__)


def test_key_order_does_not_change_hash() -> None:
    plan = _plan()
    reordered = {key: plan[key] for key in reversed(list(plan))}
    assert compute_explain_plan_hash(plan) == compute_explain_plan_hash(reordered)


def test_gamma_delta_membership_reduction_changes_hash() -> None:
    plan = _plan()
    for key, value in [("gamma_max", 0.45), ("delta_max", 0.15)]:
        changed = deepcopy(plan)
        changed[key] = value
        assert compute_explain_plan_hash(changed) != compute_explain_plan_hash(plan)
    changed = deepcopy(plan)
    changed["gamma_components"]["d_mu"] = 0.1
    assert compute_explain_plan_hash(changed) != compute_explain_plan_hash(plan)
    changed = deepcopy(plan)
    changed["reduction_weights"]["source_loss"] = 0.1
    assert compute_explain_plan_hash(changed) != compute_explain_plan_hash(plan)


def test_canonicalization_is_compact_and_stable() -> None:
    assert canonicalize_explain_plan(_plan()) == canonicalize_explain_plan(_plan())
