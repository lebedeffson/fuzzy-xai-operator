from __future__ import annotations

import numpy as np
import pytest

from fuzzyxai.replay import DelayedLabelStore, evaluate_rollback, in_canary, stream_events
from fuzzyxai.rule_effects_v2 import ConditionalRule, match_controls, within_stratum_resample
from fuzzyxai.rule_effects_v2.nonrefit import nonrefit_effect


class PlainModel:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return (values[:, 0] > 0).astype(int)


def rule(rule_id: str, support: float = 0.1) -> ConditionalRule:
    return ConditionalRule(rule_id, (0,), support, 0.2, 0.9, 1, 2, 2, 100)


def test_nonrefit_refuses_to_invent_model_ablation() -> None:
    values = np.asarray([[-1.0], [1.0]])
    with pytest.raises(TypeError, match="predict_without_rule"):
        nonrefit_effect(PlainModel(), rule("r"), values, np.asarray([0, 1]), metric=lambda y, p: float(np.mean(y == p)))


def test_conditional_resampling_stays_inside_strata() -> None:
    values = np.asarray([[1.0], [2.0], [10.0], [20.0]])
    result = within_stratum_resample(values, (0,), ("a", "a", "b", "b"), seed=7)
    assert set(result[:2, 0]) == {1.0, 2.0}
    assert set(result[2:, 0]) == {10.0, 20.0}


def test_matched_controls_are_development_only_and_at_least_five() -> None:
    candidate = rule("candidate")
    controls = [rule(f"control-{index}", support=0.1 + index / 1000) for index in range(7)]
    matched = match_controls(candidate, controls)
    assert len(matched.controls) == 5
    assert matched.selected_on == "development"


def test_replay_is_deterministic_and_delayed_labels_are_sealed() -> None:
    left = tuple(stream_events(4, seed=9))
    right = tuple(stream_events(4, seed=9))
    assert left == right
    store = DelayedLabelStore()
    store.register("x", current_index=1, delay=2, model_error=True)
    assert store.open_available(2) == ()
    assert store.open_available(3)[0].model_error


def test_canary_and_rollback_are_deterministic() -> None:
    assert in_canary("event-1", 0.1) == in_canary("event-1", 0.1)
    result = evaluate_rollback(false_block_rate=0.02, review_rate=0.2, route_fault_rate=0.1, calibration_deterioration=0.0)
    assert result.rollback
    assert "FALSE_BLOCK_CEILING_EXCEEDED" in result.reason_codes
