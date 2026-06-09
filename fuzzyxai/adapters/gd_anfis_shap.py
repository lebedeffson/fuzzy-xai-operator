from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fuzzyxai.core.explanation_object import ExplanationObject, Rule, Trace
from fuzzyxai.hierarchy.f0 import F0
from fuzzyxai.sdk import BaseAdapter, ExplanationArtifact

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = ROOT / 'data' / 'article_fixtures' / 'gd_anfis_shap_output.json'
REPORT_PATH = ROOT / 'reports' / 'chapter5' / 'scenario_reports' / 'gd_anfis_shap_action_report.md'


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class GDANFISSHAPAdapter(BaseAdapter):
    registry_id = 'gd_anfis_shap'
    adapter_class = 'GDANFISSHAPAdapter'
    input_types = ['gd_anfis_rules', 'shap_values', 'tabular_fixture']
    claim_scope = 'fixture adapter route only; excluded from quantitative claims until source repository is pinned'

    def load_fixture(self, path: str | Path = DEFAULT_FIXTURE) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding='utf-8'))

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        base = super().validate_input(payload)
        if not base['ok']:
            return base
        data = payload.get('payload', {})
        errors: list[str] = []
        for key in ['gd_anfis_rules', 'rule_activations', 'shap_values', 'rule_uncertainty', 'run_params']:
            if key not in data:
                errors.append(f'missing {key}')
        return {'ok': not errors, 'status': 'ok' if not errors else 'error', 'errors': errors, 'warnings': []}

    def explain(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = dict((payload or {}).get('payload') or self.load_fixture())
        validation = self.validate_input({'payload': data})
        if not validation['ok']:
            return {'registry_id': self.registry_id, 'status': 'error', 'validation': validation}

        probability = _clip01(float(data.get('model_output', {}).get('probability', 0.5)))
        rules = [
            Rule(str(r['id']), {'gd_anfis_condition': str(r['condition'])}, str(r['conclusion']))
            for r in data['gd_anfis_rules']
        ]
        activations = {str(k): _clip01(v) for k, v in data['rule_activations'].items()}
        uncertainties = [float(v) for v in data['rule_uncertainty'].values()]
        uncertainty = _clip01(sum(uncertainties) / max(1, len(uncertainties)))
        fixture_sha = sha256_file(DEFAULT_FIXTURE)
        trace = Trace(
            id=str(data.get('sample_id', 'gd_anfis_fixture')),
            version=str(data.get('run_params', {}).get('adapter_version', '1.0')),
            timestamp=datetime.now(timezone.utc).isoformat(),
            params=dict(data.get('run_params', {})),
            source=str(data.get('source_repo', 'source-not-provided:gd_anfis_shap')),
            checksum=fixture_sha,
        )
        explanation = ExplanationObject(
            terms={'gd_anfis_rule', 'shap_attribution', 'tabular_risk'},
            representation=F0(lambda _x, val=probability: val, label='gd_anfis_shap_risk'),
            rules=rules,
            activations=activations,
            uncertainty=uncertainty,
            trace=trace,
            reduction_loss=0.08,
            metadata={
                'adapter': self.adapter_class,
                'source_status': data.get('status', 'source-pending'),
                'fixture_sha256': fixture_sha,
                'R_k': [r.signature() for r in rules],
                'alpha_k': activations,
                'eta_k': dict(data['shap_values']),
                'u_k': dict(data['rule_uncertainty']),
                'tau_k': trace.as_dict(),
                'quantitative_claims': False,
            },
        )
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text('\n'.join([
            '# GD-ANFIS/SHAP action report',
            '',
            f"- registry_id: `{self.registry_id}`",
            f"- sample_id: `{trace.id}`",
            f"- source_status: `{data.get('status', 'source-pending')}`",
            f"- fixture_sha256: `{fixture_sha}`",
            f"- R_k: `{len(rules)}` rules",
            f"- alpha_k: `{activations}`",
            f"- eta_k / SHAP: `{data['shap_values']}`",
            f"- u_k mean: `{uncertainty:.4f}`",
            '- action: `audit_report`',
            '',
            'Сценарий исполняемый как локальный fixture-адаптер, но не используется как количественное доказательство до закрепления внешнего источника.',
        ]), encoding='utf-8')
        return {
            'registry_id': self.registry_id,
            'status': 'ok',
            'output_type': 'ExplanationArtifact + report',
            'sample_id': trace.id,
            'source_repo': data.get('source_repo', ''),
            'source_commit': data.get('source_commit', ''),
            'fixture_sha256': fixture_sha,
            'has_explanation_object': True,
            'has_diagnostic_state': False,
            'explanation_object': explanation,
            'channels': explanation.metadata,
            'report_path': str(REPORT_PATH.relative_to(ROOT)),
            'action': 'audit_report',
        }

    def to_explanation_artifact(self, raw_result: dict[str, Any]) -> ExplanationArtifact:
        payload = dict(raw_result)
        payload.pop('explanation_object', None)
        return ExplanationArtifact(
            registry_id=self.registry_id,
            artifact_type='ExplanationArtifact',
            has_explanation_object=bool(raw_result.get('has_explanation_object')),
            has_diagnostic_state=bool(raw_result.get('has_diagnostic_state')),
            payload=payload,
            trace={
                'adapter': self.adapter_class,
                'fixture_sha256': raw_result.get('fixture_sha256', ''),
                'claim_scope': self.claim_scope,
            },
            report_path=str(raw_result.get('report_path', '')),
        )


def run_fixture() -> dict[str, Any]:
    return GDANFISSHAPAdapter().explain()
