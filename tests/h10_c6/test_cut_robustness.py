from __future__ import annotations

from fuzzyxai.repair import select_global_minimum_cut
from fuzzyxai.robustness.cut_stability import _case, _jaccard, _perturbed_actions, _scenario_cost


def test_jaccard_is_order_invariant() -> None:
    assert _jaccard(("a", "b"), ("b", "a")) == 1.0


def test_cost_scaling_preserves_feasibility() -> None:
    case = _case(1)
    base = select_global_minimum_cut(case.actions, case.obligations, _scenario_cost("nominal"))
    for perturbation in ("cost_minus_0.10", "cost_plus_0.10"):
        selected = select_global_minimum_cut(
            _perturbed_actions(case.actions, perturbation),
            case.obligations,
            _scenario_cost(perturbation),
        )
        assert base.feasible and selected.feasible
        assert set(selected.covered_obligations) == set(case.obligations)


def test_equivalent_order_does_not_change_cut_set() -> None:
    case = _case(2)
    base = select_global_minimum_cut(case.actions, case.obligations, _scenario_cost("nominal"))
    changed = select_global_minimum_cut(
        _perturbed_actions(case.actions, "equivalent_node_order"),
        case.obligations,
        _scenario_cost("equivalent_node_order"),
    )
    assert frozenset(base.action_ids) == frozenset(changed.action_ids)


def test_irrelevant_evidence_cannot_cover_registered_obligations() -> None:
    case = _case(3)
    selected = select_global_minimum_cut(
        _perturbed_actions(case.actions, "irrelevant_valid_evidence"),
        case.obligations,
        _scenario_cost("irrelevant_valid_evidence"),
    )
    assert "irrelevant-evidence" not in selected.action_ids
