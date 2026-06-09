from __future__ import annotations

import json
from pathlib import Path

from fuzzyxai.adapters.gis_integro import GISIntegroAdapter


def test_gis_integro_adapter_uses_gd_anfis_shap_channels() -> None:
    raw = GISIntegroAdapter().explain()
    assert raw['status'] == 'ok'
    assert raw['registry_id'] == 'gis_integro'
    assert raw['has_explanation_object'] is True
    channels = raw['channels']
    assert channels['adapter'] == 'GISIntegroAdapter'
    assert channels['source_status'] == 'source-pending'
    assert channels['R_k']
    assert channels['alpha_k']
    assert channels['eta_k']
    assert channels['u_k']
    assert channels['gamma_route'] == 0.2
    assert channels['Delta'] == 0.08
    assert channels['quantitative_claims'] is False
    metrics_path = Path('reports/chapter5/gis_integro_route_metrics.json')
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    assert metrics['gamma_route'] == 0.2
    assert metrics['Delta'] == 0.08
