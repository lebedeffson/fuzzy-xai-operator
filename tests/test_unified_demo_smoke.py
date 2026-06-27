from __future__ import annotations

from apps.unified_demo import _read_json


def test_unified_demo_can_read_reports_json():
    data = _read_json('reports/chapter5/chapter5_experiments.json')
    assert 'calibration' in data
