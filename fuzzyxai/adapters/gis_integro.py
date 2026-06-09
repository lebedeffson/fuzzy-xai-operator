from __future__ import annotations

from pathlib import Path
from typing import Any

from fuzzyxai.adapters.gd_anfis_shap import GDANFISSHAPAdapter

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = ROOT / 'data' / 'article_fixtures' / 'gis_integro_output.json'
REPORT_PATH = ROOT / 'reports' / 'chapter5' / 'scenario_reports' / 'gis_integro_action_report.md'


class GISIntegroAdapter(GDANFISSHAPAdapter):
    registry_id = 'gis_integro'
    adapter_class = 'GISIntegroAdapter'
    input_types = ['gis_features', 'gd_anfis_rules', 'shap_regularization']
    claim_scope = 'executable GIS fixture through GD-ANFIS/SHAP channels; source remains pending'

    def load_fixture(self, path: str | Path = DEFAULT_FIXTURE) -> dict[str, Any]:
        return super().load_fixture(path)

    def explain(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        raw = super().explain(payload or {'payload': self.load_fixture()})
        if raw.get('status') != 'ok':
            return raw
        raw['registry_id'] = self.registry_id
        raw['report_path'] = str(REPORT_PATH.relative_to(ROOT))
        raw['source_repo'] = raw.get('source_repo') or 'source-not-provided:gis_integro'
        channels = dict(raw.get('channels', {}))
        channels['adapter'] = self.adapter_class
        channels['source_status'] = 'source-pending'
        channels['quantitative_claims'] = False
        raw['channels'] = channels
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text('\n'.join([
            '# GIS INTEGRO action report', '',
            f'- registry_id: `{self.registry_id}`',
            '- adapter_called: `True`',
            '- under_the_hood: `GD-ANFIS rules + SHAP regularization`',
            f"- fixture_sha256: `{raw.get('fixture_sha256')}`",
            f"- alpha_k: `{channels.get('alpha_k')}`",
            f"- eta_k / SHAP: `{channels.get('eta_k')}`",
            '- action: `audit_report`', '',
            'GIS INTEGRO исполняется как fixture-маршрут через GD-ANFIS/SHAP-каналы, но остаётся `source-pending` до закрепления внешнего источника.',
        ]), encoding='utf-8')
        return raw


def run_fixture() -> dict[str, Any]:
    return GISIntegroAdapter().explain()
