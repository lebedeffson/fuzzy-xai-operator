from __future__ import annotations

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
    assert channels['quantitative_claims'] is False
