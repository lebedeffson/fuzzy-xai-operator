from __future__ import annotations

from decimal import Decimal

import pytest

from h10_c3.baseline_methods import BASELINES, run_baseline
from h10_c3.cost_registry import CostRegistry, apply_registry, cost_cache_key
from h10_c3.fuzzy_method import run_fuzzyxai
from h10_c3.generator import generate_cases, stable_case_hash
from h10_c3.oracle import derive_gold
from h10_c3.scoring import score
from h10_c3.runner import REPO_ROOT

SCALE_FACTORS = (0.01, 0.1, 0.8, 1.0, 1.2, 10.0, 100.0)


@pytest.fixture(scope="module")
def representative_cases():
    cases = generate_cases("development", 30, 231004)
    return tuple(
        next(
            case
            for case in cases
            if case.stratum == stratum and case.repairable
        )
        for stratum in ("S1", "S2", "S3", "S4", "S5")
    )


@pytest.mark.parametrize("factor", SCALE_FACTORS)
def test_global_scale_preserves_gold_and_method_outputs(
    representative_cases,
    factor: float,
) -> None:
    for case in representative_cases:
        base_registry = CostRegistry.from_case(case)
        scaled_case = apply_registry(case, base_registry.global_scale(factor))
        base_gold = derive_gold(case)
        scaled_gold = derive_gold(scaled_case)
        assert scaled_gold.optimal_cuts == base_gold.optimal_cuts
        assert scaled_gold.optimal_cost == pytest.approx(
            factor * base_gold.optimal_cost,
            rel=1e-10,
            abs=1e-10,
        )

        base_fuzzy = run_fuzzyxai(case.method_view())
        scaled_fuzzy = run_fuzzyxai(scaled_case.method_view())
        assert scaled_fuzzy.cut == base_fuzzy.cut
        base_fuzzy_score = score(case, base_gold, base_fuzzy)
        scaled_fuzzy_score = score(scaled_case, scaled_gold, scaled_fuzzy)
        assert (
            scaled_fuzzy_score["optimal_set_membership"]
            == base_fuzzy_score["optimal_set_membership"]
        )
        assert scaled_fuzzy_score["normalized_cost_regret"] == pytest.approx(
            base_fuzzy_score["normalized_cost_regret"],
            rel=1e-10,
            abs=1e-10,
        )

        for baseline in BASELINES:
            base_result = run_baseline(baseline, case.method_view())
            scaled_result = run_baseline(baseline, scaled_case.method_view())
            assert scaled_result.cut == base_result.cut
            base_score = score(case, base_gold, base_result)
            scaled_score = score(scaled_case, scaled_gold, scaled_result)
            assert (
                scaled_score["optimal_set_membership"]
                == base_score["optimal_set_membership"]
            )
            assert scaled_score["normalized_cost_regret"] == pytest.approx(
                base_score["normalized_cost_regret"],
                rel=1e-10,
                abs=1e-10,
            )
            if base_score["raw_cost_regret"] is not None:
                assert scaled_score["raw_cost_regret"] == pytest.approx(
                    factor * base_score["raw_cost_regret"],
                    rel=1e-10,
                    abs=1e-10,
                )


def test_global_scale_covers_every_cost_component(representative_cases) -> None:
    registry = CostRegistry.from_case(representative_cases[2])
    scaled = registry.global_scale(Decimal("1.2"))
    for group in (
        "atom_costs",
        "action_costs",
        "rollback_costs",
        "human_approval_costs",
        "fixed_costs",
    ):
        before = getattr(registry, group)
        after = getattr(scaled, group)
        assert set(before) == set(after)
        assert all(after[key] == before[key] * Decimal("1.2") for key in before)


def test_cost_registry_is_part_of_cache_key(representative_cases) -> None:
    case = representative_cases[2]
    base = CostRegistry.from_case(case)
    scaled = base.global_scale("1.2")
    common = {
        "case_sha256": stable_case_hash(case),
        "method_sha256": "method",
        "protocol_sha256": "protocol",
        "solver_config_sha256": "solver",
    }
    base_key = cost_cache_key(cost_registry_sha256=base.sha256, **common)
    scaled_key = cost_cache_key(cost_registry_sha256=scaled.sha256, **common)
    assert base.sha256 != scaled.sha256
    assert base_key != scaled_key


def test_sensitivity_does_not_reselect_frozen_baseline(
    representative_cases,
) -> None:
    selection_path = (
        REPO_ROOT
        / "artifacts"
        / "h10_c3_v23"
        / "lock"
        / "baseline_selection.json"
    )
    before = selection_path.read_bytes()
    case = representative_cases[2]
    scaled_case = apply_registry(
        case,
        CostRegistry.from_case(case).global_scale("1.2"),
    )
    assert run_baseline("weighted_greedy", scaled_case.method_view()).method == (
        "weighted_greedy"
    )
    assert selection_path.read_bytes() == before


@pytest.mark.parametrize("factor", (0, -1))
def test_global_scale_rejects_nonpositive_factor(
    representative_cases,
    factor: float,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        CostRegistry.from_case(representative_cases[0]).global_scale(factor)
