from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'fuzzyxai_experiments' / 'reports'


def _read(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding='utf-8'))


def test_fuzzyxai_experiments_run_all_outputs_required_reports() -> None:
    subprocess.run(['bash', 'fuzzyxai_experiments/run_all.sh'], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    expected = {
        'ch2_bc_results.json',
        'ch2_synthesis.json',
        'ch2_critical_ruptures.json',
        'ch3_selection.json',
        'ch3_reduction.json',
        'ch3_diagnostic_stand.json',
        'ch4_integration.json',
        'ch5_scenario_runs.json',
        'ch5_hybrid.json',
        'ch5_gis.json',
        'ch5_beacon.json',
    }
    assert expected <= {p.name for p in REPORTS.glob('*.json')}


def test_fuzzyxai_experiments_key_chapter5_numbers_are_from_reports() -> None:
    gis = _read('ch5_gis.json')
    assert gis['metrics']['gamma_route'] == 0.2
    assert gis['metrics']['Delta'] == 0.08

    beacon = _read('ch5_beacon.json')
    assert beacon['route_supported'] == 83
    assert beacon['baseline_alerts_before'] == 64
    assert beacon['fuzzyxai_alerts_after'] == 11
    assert beacon['audit_reports'] == 12
    assert beacon['fixture_sha256']

    hybrid = _read('ch5_hybrid.json')
    assert hybrid['n_images'] == 1000
    assert hybrid['fuzzyxai_missed_critical_quality_cases'] == 0
    assert hybrid['baseline_missed_critical_quality_cases'] > 0


def test_fuzzyxai_experiments_core_smoke() -> None:
    from fuzzyxai_experiments.src.fuzzyxai_core import ExplainPlan, FuzzyExplanation, Rule, Trace, compose, synthesize_alignment
    from fuzzyxai_experiments.src.risk_observer import RiskObserver

    plan = ExplainPlan.load('fuzzyxai_experiments/explain_plans/bc_plan.yaml')
    e1 = FuzzyExplanation({'low', 'high'}, {'high': 0.7}, [Rule('r1', 'risk high', 'audit')], {'r1': 0.7}, 0.2, Trace('model', 'v1', 't'))
    e2 = FuzzyExplanation({'low', 'high'}, {'high': 0.72}, [Rule('r1', 'risk high', 'audit')], {'r1': 0.72}, 0.2, Trace('model', 'v1', 't'))
    assert e1.compute_d_E(e2, plan) >= 0
    assert not isinstance(compose(e1, e2, {'gamma_max': 0.4}, plan), str)
    synth = synthesize_alignment(e1, e2, ExplainPlan(path='inline', data={'alignment_synthesis': {'gamma_max': 0.4, 'candidates': [{'name': 'identity'}]}}))
    assert isinstance(synth, dict)
    decision = RiskObserver({'predicted_risk': 0.5, 'uncertainty': 0.2, 'interpretability_gap': 0.1, 'reduction_loss': 0.1, 'diagnostic': 0.1}, [0.12, 0.28, 0.52, 0.8]).decide(e1, 0.1)
    assert 'action' in decision
