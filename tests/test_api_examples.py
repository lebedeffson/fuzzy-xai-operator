from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_openapi_has_required_endpoints() -> None:
    spec = yaml.safe_load((ROOT / 'api/openapi.yaml').read_text(encoding='utf-8'))
    assert '/v1/explain' in spec['paths']
    assert '/v1/risk-action' in spec['paths']


def test_api_examples_match_contract_shape() -> None:
    explain_req = json.loads((ROOT / 'api/example_requests/explain_request.json').read_text(encoding='utf-8'))
    explain_resp = json.loads((ROOT / 'api/example_responses/explain_response.json').read_text(encoding='utf-8'))
    risk_req = json.loads((ROOT / 'api/example_requests/risk_action_request.json').read_text(encoding='utf-8'))
    risk_resp = json.loads((ROOT / 'api/example_responses/risk_action_response.json').read_text(encoding='utf-8'))
    assert {'registry_id', 'input_type', 'payload', 'explain_plan_hash'} <= set(explain_req)
    assert explain_resp['artifact_type'] == 'ExplanationArtifact'
    assert {'artifact_path', 'observer_mode'} <= set(risk_req)
    assert {'chi_R', 'chi_Auto', 'rho', 'action'} <= set(risk_resp)
