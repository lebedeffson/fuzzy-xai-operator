from __future__ import annotations

import json
from pathlib import Path

from experiments.chapter5_evidence_integrity import audit_repository, inspect_case


def make_case(root: Path, scenario: str, *, action='audit_report') -> Path:
    folder = root / 'applications' / 'scenarios' / scenario
    result = {'action': action, 'diagnostic_id': ''}
    route = {'scenario_id': scenario, 'computed_result': result, 'diagnostics': [], 'final_action': action, 'source_commit': 'a' * 40}
    trace = {'scenario_id': scenario, 'package_type': 'FuzzyXAIProofTrace', 'computed_result': result, 'route': route, 'final_action': action}
    site = {'scenario_id': scenario, 'final_action': action, 'diagnostic_id': '', 'verifier_status': 'PASS'}
    for relative, obj in [('route/route.json', route), ('proof/proof_trace.json', trace), ('site_payload/scenario.json', site)]:
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding='utf-8')
    return folder


def test_consistent_artifacts_pass(tmp_path: Path) -> None:
    folder = make_case(tmp_path, 'medical_ecg_signal')
    result = inspect_case(folder, 'medical_ecg_signal')
    assert result['status'] == 'PASS', result


def test_tampered_site_action_fails(tmp_path: Path) -> None:
    folder = make_case(tmp_path, 'medical_ecg_signal')
    path = folder / 'site_payload/scenario.json'
    obj = json.loads(path.read_text())
    obj['final_action'] = 'accept'
    path.write_text(json.dumps(obj))
    result = inspect_case(folder, 'medical_ecg_signal')
    assert 'final_action_mismatch' in result['errors']


def test_missing_proof_does_not_silently_pass(tmp_path: Path) -> None:
    folder = make_case(tmp_path, 'medical_ecg_signal')
    (folder / 'proof/proof_trace.json').unlink()
    result = inspect_case(folder, 'medical_ecg_signal')
    assert result['status'] == 'FAIL'


def test_audit_no_placeholder_metrics(tmp_path: Path) -> None:
    for sid in ('medical_ecg_signal', 'hybrid_xiris', 'gd_anfis_shap', 'beacon_xai', 'gis_integro'):
        make_case(tmp_path, sid)
    result = audit_repository(tmp_path, tmp_path / 'out', strict=True)
    assert result['pass']
    assert result['failures'] == 0
    assert result['clinical_validation'] is False
    assert (tmp_path / 'out/integrity_summary.json').is_file()


def test_tampered_proof_computation_is_detected(tmp_path: Path) -> None:
    folder = make_case(tmp_path, 'medical_ecg_signal')
    path = folder / 'proof/proof_trace.json'
    obj = json.loads(path.read_text())
    obj['computed_result']['action'] = 'block'
    path.write_text(json.dumps(obj))
    result = inspect_case(folder, 'medical_ecg_signal')
    assert result['status'] == 'FAIL'
    assert 'route_vs_proof_computed_mismatch' in result['errors']
