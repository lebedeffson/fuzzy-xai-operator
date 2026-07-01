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
DEFAULT_FIXTURE = ROOT / 'data' / 'article_fixtures' / 'beacon_xai_output.json'
REPORT_PATH = ROOT / 'reports' / 'chapter5' / 'scenario_reports' / 'beacon_xai_action_report.md'


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class BeaconXAIAdapter(BaseAdapter):
    registry_id = 'beacon_xai'
    adapter_class = 'BeaconXAIAdapter'
    input_types = ['beacon_report', 'feature_attributions', 'counterfactual_stability']
    claim_scope = 'pinned repository fixture route; no local model retraining or quantitative benchmark claim'

    def load_fixture(self, path: str | Path = DEFAULT_FIXTURE) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding='utf-8'))

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        base = super().validate_input(payload)
        if not base['ok']:
            return base
        data = payload.get('payload', {})
        required = ['beacon_score', 'feature_attributions', 'rules', 'rule_activations', 'uncertainty', 'run_params']
        errors = [f'missing {key}' for key in required if key not in data]
        return {'ok': not errors, 'status': 'ok' if not errors else 'error', 'errors': errors, 'warnings': []}

    def explain(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = dict((payload or {}).get('payload') or self.load_fixture())
        validation = self.validate_input({'payload': data})
        if not validation['ok']:
            return {'registry_id': self.registry_id, 'status': 'error', 'validation': validation}
        score = _clip01(data['beacon_score'])
        rules = [Rule(str(r['id']), {'beacon_condition': str(r['condition'])}, str(r['conclusion'])) for r in data['rules']]
        activations = {str(k): _clip01(v) for k, v in data['rule_activations'].items()}
        uncertainty = _clip01(sum(float(v) for v in data['uncertainty'].values()) / max(1, len(data['uncertainty'])))
        fixture_sha = sha256_file(DEFAULT_FIXTURE)
        trace = Trace(
            id=str(data.get('sample_id', 'beacon_fixture')),
            version=str(data.get('run_params', {}).get('adapter_version', '1.0')),
            timestamp=datetime.now(timezone.utc).isoformat(),
            params=dict(data.get('run_params', {})),
            source=str(data.get('source_repo', 'https://github.com/fims9000/BeaconXAI')),
            checksum=fixture_sha,
        )
        explanation = ExplanationObject(
            terms={'beacon_score', 'counterfactual_stability', 'feature_attribution'},
            representation=F0(lambda _x, val=score: val, label='beacon_xai_score'),
            rules=rules,
            activations=activations,
            uncertainty=uncertainty,
            trace=trace,
            reduction_loss=0.10,
            metadata={
                'adapter': self.adapter_class,
                'source_repo': data.get('source_repo'),
                'source_commit': data.get('source_commit'),
                'fixture_sha256': fixture_sha,
                'R_k': [r.signature() for r in rules],
                'alpha_k': activations,
                'eta_k': dict(data['feature_attributions']),
                'u_k': dict(data['uncertainty']),
                'tau_k': trace.as_dict(),
                'quantitative_claims': False,
            },
        )
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text('\n'.join([
            '# BEACON-XAI action report', '',
            f'- registry_id: `{self.registry_id}`',
            f"- source_repo: `{data.get('source_repo')}`",
            f"- source_commit: `{data.get('source_commit')}`",
            f'- fixture_sha256: `{fixture_sha}`',
            f"- alpha_k: `{activations}`",
            f"- eta_k: `{data['feature_attributions']}`",
            f'- u_k mean: `{uncertainty:.4f}`',
            '- action: `audit_report`', '',
            'BEACON-XAI подключён как pinned fixture route. Количественные claims не поднимаются без отдельного benchmark-протокола.',
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
        return ExplanationArtifact(self.registry_id, 'ExplanationArtifact', bool(raw_result.get('has_explanation_object')), bool(raw_result.get('has_diagnostic_state')), payload, {'adapter': self.adapter_class, 'claim_scope': self.claim_scope}, str(raw_result.get('report_path', '')))


def run_fixture() -> dict[str, Any]:
    return BeaconXAIAdapter().explain()
