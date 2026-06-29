from __future__ import annotations

from fuzzyxai import Candidate, select_minimal_sufficient


def _candidates():
    return [
        Candidate('F0', {'u_num', 'u_ling'}, 0.10, 0.03, 0.42),
        Candidate('FI', {'u_num', 'u_ling', 'u_int'}, 0.22, 0.05, 0.30),
        Candidate('FH', {'u_num', 'u_ling', 'u_exp'}, 0.30, 0.81, 0.25),
        Candidate('FNsrc', {'u_num', 'u_ling', 'u_if', 'u_conf', 'u_trace'}, 0.42, 0.09, 0.18),
        Candidate('FML-audit', {'u_num', 'u_ling', 'u_int', 'u_exp', 'u_conf', 'u_trace', 'u_multi'}, 0.68, 1.00, 0.04),
    ]


def test_every_operational_uncertainty_type_is_covered():
    candidates = _candidates()
    universe = {'u_num', 'u_ling', 'u_int', 'u_exp', 'u_conf', 'u_trace', 'u_multi'}
    covered = set().union(*(c.coverage for c in candidates))
    assert universe <= covered


def test_minimal_class_exists_for_each_atomic_type():
    candidates = _candidates()
    for u in ['u_num', 'u_int', 'u_exp', 'u_conf']:
        admissible = [c for c in candidates if u in c.coverage]
        assert admissible
        minimal = min(admissible, key=lambda c: (c.cognitive_complexity, c.expected_reduction_loss))
        assert u in minimal.coverage


def test_removing_interval_loses_minimal_u_int_coverage():
    candidates = [c for c in _candidates() if c.name != 'FI']
    selected = select_minimal_sufficient({'u_num', 'u_ling', 'u_int'}, candidates, 'user')
    assert selected.name == 'FML-audit'


def test_removing_hesitant_loses_minimal_expert_coverage():
    candidates = [c for c in _candidates() if c.name != 'FH']
    selected = select_minimal_sufficient({'u_num', 'u_ling', 'u_exp'}, candidates, 'user')
    assert selected.name == 'FML-audit'


def test_uncovered_profile_returns_choice_diagnostic():
    result = select_minimal_sufficient({'u_ethical'}, _candidates(), 'audit')
    assert getattr(result, 'code', '') == 'D_choice'
