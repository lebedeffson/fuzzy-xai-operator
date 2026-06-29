from __future__ import annotations

from apps.services.layered_case import LayeredCaseService, build_case_state
from fuzzyxai.studio.operators import build_operator_trace


def test_operator_trace_contains_core_operators() -> None:
    service = LayeredCaseService.create()
    st = build_case_state(service, 'safe', sample_index=0, dataset_mode='breast_cancer')
    rows = build_operator_trace(st)
    names = {r['operator'] for r in rows}
    assert 'OmegaOperator' in names
    assert 'RiskObserver' in names
    assert 'ContextTopos' in names
    assert 'ActionPolicy' in names


def test_operator_trace_has_inputs_outputs_status() -> None:
    service = LayeredCaseService.create()
    st = build_case_state(service, 'rupture', sample_index=113, dataset_mode='breast_cancer')
    rows = build_operator_trace(st)
    for r in rows:
        assert r.get('takes_from')
        assert r.get('outputs')
        assert r.get('status') is not None
        assert r.get('severity') in {'ok', 'warning', 'critical'}
        assert bool(r.get('signal'))
