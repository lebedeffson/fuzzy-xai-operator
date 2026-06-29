from __future__ import annotations

from fuzzyxai import F0, HesitantFS, IntervalFS, MultiLevelFS, NeutrosophicFS

RANKS = {'F0': 0, 'FI': 1, 'FH': 1, 'FNsrc': 2, 'FML': 3}


def _rank(level):
    return RANKS.get(getattr(level, 'class_name', 'F0'), 0)


def _simulate_steps(obj: MultiLevelFS, delta_max: float):
    levels = list(obj.levels)
    steps = 0
    while True:
        changed = False
        for i, level in enumerate(levels):
            if _rank(level) == 0:
                continue
            red, delta = level.reduce_to_f0()
            if delta <= delta_max:
                levels[i] = red
                steps += 1
                changed = True
                break
        if not changed:
            return MultiLevelFS(levels, gamma=set(), weights=obj.weights), steps


def _object():
    return MultiLevelFS([
        IntervalFS(lambda x: 0.45, lambda x: 0.55),
        HesitantFS(lambda x: [0.48, 0.52]),
        NeutrosophicFS(lambda x: 0.5, lambda x: 0.0, lambda x: 0.5),
    ])


def test_multilevel_reduction_terminates_within_m_times_n():
    final, steps = _simulate_steps(_object(), delta_max=1.0)
    assert steps <= len(final.levels) * max(RANKS.values())


def test_after_stop_no_level_has_admissible_reduction():
    final, _ = _simulate_steps(_object(), delta_max=0.0)
    for level in final.levels:
        if _rank(level) > 0:
            _, delta = level.reduce_to_f0()
            assert delta > 0.0


def test_zero_threshold_reduces_only_lossless_levels():
    final, _ = _simulate_steps(_object(), delta_max=0.0)
    assert any(not isinstance(level, F0) for level in final.levels)


def test_large_threshold_reaches_simple_levels():
    final, _ = _simulate_steps(_object(), delta_max=1.0)
    assert all(isinstance(level, F0) for level in final.levels)
