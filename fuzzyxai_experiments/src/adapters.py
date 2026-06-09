from __future__ import annotations

from pathlib import Path
from typing import Any

from fuzzyxai_experiments.src.utils import read_json, sha256_file


def gd_anfis_shap_adapter(X: Any = None, model: Any = None, shap_values: Any = None) -> dict[str, Any]:
    """Run GD-ANFIS/SHAP fixture adapter and return E_k/report payload."""
    fixture = read_json('data/article_fixtures/gd_anfis_shap_output.json')
    return {'status': fixture['status'], 'registry_id': 'gd_anfis_shap', 'R_k': fixture['gd_anfis_rules'], 'alpha_k': fixture['rule_activations'], 'eta_k': fixture['shap_values'], 'u_k': fixture['rule_uncertainty'], 'tau_k': fixture['run_params'], 'fixture_sha256': sha256_file('data/article_fixtures/gd_anfis_shap_output.json')}


def gis_integro_adapter(raster_data: Any = None, attributes: Any = None) -> dict[str, Any]:
    """Run GIS INTEGRO fixture adapter and return route metrics."""
    fixture = read_json('data/article_fixtures/gis_integro_output.json')
    metrics = read_json('reports/chapter5/gis_integro_route_metrics.json')
    return {'status': fixture['status'], 'registry_id': 'gis_integro', 'R_k': fixture['gd_anfis_rules'], 'alpha_k': fixture['rule_activations'], 'eta_k': fixture['shap_values'], 'u_k': fixture['rule_uncertainty'], 'tau_k': fixture['run_params'], 'gamma_route': metrics['gamma_route'], 'Delta': metrics['Delta'], 'fixture_sha256': metrics['fixture_sha256']}


def beacon_xai_adapter(payload: Any = None) -> dict[str, Any]:
    """Run BEACON-XAI fixture adapter."""
    fixture = read_json('data/article_fixtures/beacon_xai_output.json')
    return {'status': fixture['status'], 'registry_id': 'beacon_xai', 'source_repo': fixture['source_repo'], 'source_commit': fixture['source_commit'], 'R_k': fixture['rules'], 'alpha_k': fixture['rule_activations'], 'eta_k': fixture['feature_attributions'], 'u_k': fixture['uncertainty'], 'tau_k': fixture['run_params'], 'fixture_sha256': sha256_file('data/article_fixtures/beacon_xai_output.json')}


def hybrid_xiris_adapter(image_path: str | Path | None = None) -> dict[str, Any]:
    """Return HYBRID-XIRIS blocking fixture."""
    return read_json('reports/chapter5/hybrid_xiris_blocking_case.json')
