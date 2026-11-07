"""Check reproducible proof/route/site consistency for chapter 5 applications.

This verifies artifact consistency, not the accuracy of source ML models.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / 'framework' / 'fuzzyxai'
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))
SCENARIOS = ('medical_ecg_signal', 'hybrid_xiris', 'gd_anfis_shap', 'beacon_xai', 'gis_integro')


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError(f'{path.name} must contain a JSON object')
    return result


def inspect_case(directory: Path, scenario_id: str, *, recompute: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        route = _read(directory / 'route/route.json')
        trace = _read(directory / 'proof/proof_trace.json')
        site = _read(directory / 'site_payload/scenario.json')
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        return {'scenario_id': scenario_id, 'status': 'FAIL', 'errors': [str(exc)], 'warnings': warnings}
    if trace.get('package_type') != 'FuzzyXAIProofTrace':
        errors.append('invalid_proof_type')
    if any(data.get('scenario_id') != scenario_id for data in (route, trace, site)):
        errors.append('scenario_id_mismatch')
    if route.get('computed_result') != trace.get('computed_result'):
        errors.append('route_vs_proof_computed_mismatch')
    if trace.get('route', {}).get('computed_result') != trace.get('computed_result'):
        errors.append('proof_internal_computed_mismatch')
    action = route.get('final_action')
    if not action or any(action != other for other in (
        route.get('computed_result', {}).get('action'), trace.get('final_action'), site.get('final_action')
    )):
        errors.append('final_action_mismatch')
    computed_d = route.get('computed_result', {}).get('diagnostic_id', '')
    site_d = site.get('diagnostic_id', '')
    if computed_d != site_d:
        errors.append('diagnostic_id_mismatch')
    diag_ids = {d.get('diagnostic_id') for d in route.get('diagnostics', []) if isinstance(d, dict)}
    if computed_d and computed_d not in diag_ids:
        errors.append('diagnostic_not_in_route')
    if route.get('computed_result', {}).get('chi_crit') == 1 and action != 'block':
        errors.append('critical_rupture_not_blocked')
    source_commit = str(route.get('source_commit') or trace.get('source_commit') or '')
    if not re.fullmatch(r'[0-9a-f]{40}', source_commit):
        warnings.append('source_commit_not_full_sha')
    if site.get('verifier_status') != 'PASS':
        warnings.append('site_verifier_not_pass')
    if recompute:
        from fuzzyxai.examples import load_example
        from fuzzyxai.core.route import build_route
        from fuzzyxai.proof.trace import build_proof_trace
        from fuzzyxai.proof.verifier import verify_proof_trace

        current = build_route(load_example(scenario_id))
        verification = verify_proof_trace(build_proof_trace(current))
        if not verification.valid:
            errors.append('fresh_proof_failed')
        if current.final_action != action:
            errors.append('computed_action_vs_snapshot_mismatch')
        if current.computed_result.get('diagnostic_id', '') != computed_d:
            errors.append('computed_diagnostic_vs_snapshot_mismatch')
        for key in ('gamma', 'delta', 'gamma_rule_shap', 'gamma_route', 'chi_crit'):
            if key in current.computed_result and key in route.get('computed_result', {}):
                if current.computed_result[key] != route['computed_result'][key]:
                    errors.append('computed_' + key + '_vs_snapshot_mismatch')
    return {
        'scenario_id': scenario_id,
        'status': 'FAIL' if errors else ('WARN' if warnings else 'PASS'),
        'action': action,
        'diagnostic_id': computed_d,
        'errors': errors,
        'warnings': warnings,
        'source_commit': source_commit,
        'verification_scope': 'artifact_consistency_not_model_validation',
    }


def audit_repository(root: Path = ROOT, out_dir: Path | None = None, strict: bool = False, recompute: bool = False) -> dict[str, Any]:
    reports = [inspect_case(root / 'applications/scenarios' / sid, sid, recompute=recompute) for sid in SCENARIOS]
    failures = sum(row['status'] == 'FAIL' for row in reports)
    warnings = sum(row['status'] == 'WARN' for row in reports)
    summary = {
        'protocol': 'chapter5_evidence_integrity_v1',
        'cases_checked': len(reports),
        'recomputed_from_source': recompute,
        'failures': failures,
        'warnings': warnings,
        'pass': not failures and (not strict or not warnings),
        'clinical_validation': False,
        'quantitative_model_benchmark': False,
        'cases': reports,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'integrity_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        with (out_dir / 'integrity_cases.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['scenario_id', 'status', 'action', 'diagnostic_id', 'source_commit', 'errors', 'warnings'], lineterminator='\n')
            writer.writeheader()
            for case in reports:
                writer.writerow({key: '|'.join(case.get(key, [])) if key in ('errors', 'warnings') else case.get(key, '') for key in writer.fieldnames})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/chapter5/evidence_integrity')
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    summary = audit_repository(out_dir=args.out, strict=args.strict, recompute=args.recompute)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary['pass'] else 1)


if __name__ == '__main__':
    main()
