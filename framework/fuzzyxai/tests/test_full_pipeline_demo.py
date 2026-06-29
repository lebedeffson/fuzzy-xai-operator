from __future__ import annotations

from pathlib import Path

from full_pipeline_demo import run_full_pipeline


def test_full_pipeline_demo_report_created():
    report = run_full_pipeline(open_browser=False)
    assert report['status'] == 'PASS'
    assert report['composition']['index'] > 0
    assert report['risk_observer']['risk_reduction'] > 0
    assert Path('reports/full_demo/index.html').exists()
    assert Path('reports/full_demo/full_pipeline_report.json').exists()
