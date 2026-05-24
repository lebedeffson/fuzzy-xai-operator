from __future__ import annotations

from pathlib import Path

from fuzzyxai.risk.observer_pipeline import build_full_observer_pipeline_report, write_full_observer_pipeline_report


def test_full_observer_pipeline_builds_active_observer_report():
    report = build_full_observer_pipeline_report()
    obs = report['with_observer']
    assert report['status'] == 'PASS'
    assert report['thesis_layer'] == 'active risk-oriented observer over a predictive interface'
    assert 0 <= obs['risk_score'] <= 1
    assert 0 <= obs['uncertainty'] <= 1
    assert obs['selected_representation'] == 'FML-audit'
    assert obs['safe_action'] in {'accept', 'lower_confidence', 'request_more_data', 'defer_to_human', 'block'}
    assert len(obs['composition_edges']) == 2


def test_full_observer_pipeline_writes_reports():
    report = build_full_observer_pipeline_report()
    paths = write_full_observer_pipeline_report(report)
    assert Path(paths['json']).exists()
    assert Path(paths['markdown']).exists()
    assert Path(paths['html']).exists()
