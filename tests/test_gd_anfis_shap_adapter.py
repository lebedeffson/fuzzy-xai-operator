from __future__ import annotations

from pathlib import Path

from fuzzyxai.adapters.gd_anfis_shap import GDANFISSHAPAdapter, DEFAULT_FIXTURE, sha256_file
from fuzzyxai.sdk import ExplanationArtifact


def test_gd_anfis_shap_adapter_maps_required_channels() -> None:
    adapter = GDANFISSHAPAdapter()
    raw = adapter.explain()
    assert raw['status'] == 'ok'
    assert raw['has_explanation_object'] is True
    channels = raw['channels']
    assert channels['R_k']
    assert channels['alpha_k']
    assert channels['eta_k']
    assert channels['u_k']
    assert channels['tau_k']
    assert channels['fixture_sha256'] == sha256_file(DEFAULT_FIXTURE)
    assert channels['gamma_route'] == 0.233333
    assert channels['Delta'] == 0.08
    assert channels['quantitative_claims'] is False
    artifact = adapter.to_explanation_artifact(raw)
    assert isinstance(artifact, ExplanationArtifact)
    assert artifact.registry_id == 'gd_anfis_shap'
    assert artifact.report_path.endswith('gd_anfis_shap_action_report.md')
    report = Path(artifact.report_path).read_text(encoding='utf-8')
    assert 'registry_id: `gd_anfis_shap`' in report
    assert 'контрольный маршрут' in report
