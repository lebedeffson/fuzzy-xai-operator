from __future__ import annotations

from datetime import datetime, timezone

from fuzzyxai.core.alignment_synthesis import AlignmentResult, enumerate_alignment_candidates, score_alignment_candidate, synthesize_alignment, write_alignment_report
from fuzzyxai.core.diagnostics import DiagnosticState
from fuzzyxai.core.explanation_object import ExplanationObject, Rule, Trace
from fuzzyxai.hierarchy.f0 import F0


def _E(label: str, activation: float = 0.7) -> ExplanationObject:
    return ExplanationObject(
        terms={'low', 'high'},
        representation=F0(lambda _x: activation, label=label),
        rules=[Rule('r_high', {'risk': 'high'}, 'audit')],
        activations={'r_high': activation},
        uncertainty=0.2,
        trace=Trace(label, 'v1', datetime.now(timezone.utc).isoformat()),
        reduction_loss=0.05,
    )


def test_alignment_synthesis_returns_diagnostic_without_finite_space() -> None:
    result = enumerate_alignment_candidates(_E('a'), _E('b'), {})
    assert isinstance(result, DiagnosticState)
    assert result.code == 'alignment_space_not_finite'


def test_alignment_synthesis_accepts_finite_candidate(tmp_path) -> None:
    plan = {'alignment_synthesis': {'gamma_max': 0.4, 'lambda_delta': 0.2, 'candidates': [{'name': 'identity', 'term_map': {'low': 'low', 'high': 'high'}, 'rule_map': {'r_high': 'r_high'}}]}}
    result = synthesize_alignment(_E('a', 0.7), _E('b', 0.72), plan)
    assert isinstance(result, AlignmentResult)
    assert result.J <= 0.4
    score = score_alignment_candidate(result.candidate, _E('a', 0.7), _E('b', 0.72), plan)
    assert {'gamma', 'Delta_T', 'J(T)'} <= set(score)
    report = write_alignment_report(tmp_path / 'alignment.json', _E('a'), _E('b'), plan)
    assert 'T_ij' in report


def test_alignment_synthesis_blocks_high_cost() -> None:
    plan = {'alignment_synthesis': {'gamma_max': 0.01, 'candidates': [{'name': 'bad', 'term_map': {'low': 'low'}, 'rule_map': {'r_high': 'r_high'}, 'activation_scale': 0.1}]}}
    result = synthesize_alignment(_E('a', 0.9), _E('b', 0.1), plan)
    assert isinstance(result, DiagnosticState)
    assert result.code == 'alignment_cost_exceeds_gamma_max'
