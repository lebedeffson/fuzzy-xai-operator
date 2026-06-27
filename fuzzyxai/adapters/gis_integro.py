from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fuzzyxai.adapters.gd_anfis_shap import GDANFISSHAPAdapter, compute_route_metrics, sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = ROOT / 'data' / 'article_fixtures' / 'gis_integro_output.json'
REPORT_PATH = ROOT / 'reports' / 'chapter5' / 'scenario_reports' / 'gis_integro_action_report.md'
METRICS_JSON = ROOT / 'reports' / 'chapter5' / 'gis_integro_route_metrics.json'
METRICS_CSV = ROOT / 'reports' / 'chapter5' / 'gis_integro_route_metrics.csv'


class GISIntegroAdapter(GDANFISSHAPAdapter):
    registry_id = 'gis_integro'
    adapter_class = 'GISIntegroAdapter'
    input_types = ['gis_features', 'gd_anfis_rules', 'shap_regularization']
    claim_scope = 'executable GIS fixture through GD-ANFIS/SHAP channels; source remains pending'
    report_path = REPORT_PATH

    def load_fixture(self, path: str | Path = DEFAULT_FIXTURE) -> dict[str, Any]:
        return super().load_fixture(path)

    def explain(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = dict((payload or {}).get('payload') or self.load_fixture())
        raw = super().explain({'payload': data})
        if raw.get('status') != 'ok':
            return raw
        raw['registry_id'] = self.registry_id
        raw['report_path'] = str(REPORT_PATH.relative_to(ROOT))
        raw['source_repo'] = raw.get('source_repo') or 'source-not-provided:gis_integro'
        fixture_sha = sha256_file(DEFAULT_FIXTURE)
        raw['fixture_sha256'] = fixture_sha
        channels = dict(raw.get('channels', {}))
        channels['adapter'] = self.adapter_class
        channels['source_status'] = 'source-pending'
        channels['quantitative_claims'] = False
        channels['fixture_sha256'] = fixture_sha
        route_metrics = compute_route_metrics(data, reduction_loss=float(channels.get('Delta', 0.08)))
        channels['route_metrics'] = route_metrics
        channels['gamma_route'] = route_metrics['gamma_route']
        channels['Delta'] = route_metrics['Delta']
        raw['channels'] = channels
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
        metrics_payload = {
            'registry_id': self.registry_id,
            'sample_id': raw.get('sample_id'),
            'source_status': 'source-pending',
            'fixture_sha256': fixture_sha,
            **route_metrics,
            'formula': 'gamma_route = max(|p - mean(alpha_k)|, |p - positive_SHAP_support|); Delta = reduction_loss',
            'claim_scope': 'контрольный маршрут; не количественное сравнение качества исходной модели',
        }
        METRICS_JSON.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding='utf-8')
        METRICS_CSV.write_text(
            'registry_id,sample_id,probability,mean_alpha_k,positive_shap_support,gamma_route,Delta,fixture_sha256,claim_scope\n'
            f"{self.registry_id},{raw.get('sample_id')},{route_metrics['probability']},{route_metrics['mean_alpha_k']},{route_metrics['positive_shap_support']},{route_metrics['gamma_route']},{route_metrics['Delta']},{fixture_sha},контрольный маршрут; не количественное сравнение качества исходной модели\n",
            encoding='utf-8',
        )
        REPORT_PATH.write_text('\n'.join([
            '# GIS INTEGRO action report', '',
            f'- registry_id: `{self.registry_id}`',
            '- adapter_called: `True`',
            '- under_the_hood: `GD-ANFIS rules + SHAP regularization`',
            f"- fixture_sha256: `{fixture_sha}`",
            f"- alpha_k: `{channels.get('alpha_k')}`",
            f"- eta_k / SHAP: `{channels.get('eta_k')}`",
            f"- gamma_route: `{route_metrics['gamma_route']:.4f}`",
            f"- Delta: `{route_metrics['Delta']:.4f}`",
            '- action: `audit_report`', '',
            'GIS INTEGRO исполняется как контрольный fixture-маршрут через GD-ANFIS/SHAP-каналы. Числа gamma_route и Delta вычислены из fixture, но сценарий остаётся `source-pending` до закрепления внешнего источника.',
        ]), encoding='utf-8')
        return raw


def run_fixture() -> dict[str, Any]:
    return GISIntegroAdapter().explain()
