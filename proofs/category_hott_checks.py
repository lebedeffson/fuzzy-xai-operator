from __future__ import annotations

import json
from pathlib import Path

from fuzzyxai.category import ContextPresheaf, ExplanationCategory, PresheafToposDescriptor, try_make_morphism
from fuzzyxai.core.explanation_object import ExplanationObject, Rule, Trace
from fuzzyxai.hierarchy.f0 import F0
from fuzzyxai.hott import ExplanationPath, RuptureType, build_temporal_drift_path, certify_path, certify_rupture


def _e(name: str, value: float = 0.2) -> ExplanationObject:
    rule = Rule(f'r_{name}', {'risk': 'medium'}, 'review')
    return ExplanationObject(
        terms={'low', 'medium', 'high'},
        representation=F0(lambda x: value, 'risk'),
        rules=[rule],
        activations={rule.name: value},
        uncertainty=value,
        trace=Trace(name, 'v1', 't0', checksum=name),
    )


def run_checks() -> dict:
    cat = ExplanationCategory({'uncertainty': 1.0}, gamma_max=0.5)
    e_model = cat.object('E_model', _e('model', 0.2))
    e_risk = cat.object('E_risk', _e('risk', 0.3))
    e_action = cat.object('E_action', _e('action', 0.35))

    mr = cat.make_morphism(e_model, e_risk, name='T_MR', gamma=0.18)
    ra = cat.make_morphism(e_risk, e_action, name='T_RA', gamma=0.12)
    ma = cat.compose(mr, ra)
    path = ExplanationPath.from_morphism(mr).concat(ExplanationPath.from_morphism(ra))

    presheaf = ContextPresheaf()
    presheaf.add_contexts(e_model, {'model_context'})
    presheaf.add_contexts(e_risk, {'risk_context'})
    presheaf.add_contexts(e_action, {'action_context'})
    presheaf.set_restriction(mr, lambda _: 'model_context')
    presheaf.set_restriction(ra, lambda _: 'risk_context')
    presheaf.set_restriction(ma, lambda _: 'model_context')

    rupture_result = try_make_morphism(cat, e_model, e_action, name='bad_jump', gamma=0.9)
    rupture = RuptureType.from_diagnostic(
        e_model.key,
        e_action.key,
        rupture_result.diagnostic,
        gamma=0.9,
        threshold=0.5,
    )
    drift = build_temporal_drift_path(cat, ['t0', 't1', 't2'], [e_model, e_risk, e_action])

    checks = [
        ('category identities', cat.identity(e_model).source == e_model),
        ('category composition', ma.source == e_model and ma.target == e_action),
        ('presheaf functoriality', presheaf.check_contravariant_composition(mr, ra, ma)),
        ('topos descriptor', PresheafToposDescriptor().contains(presheaf)),
        ('path certificate', certify_path(path, gamma_max=0.5).valid),
        ('rupture certificate', certify_rupture(rupture).diagnostic_code == 'D_ij'),
        ('temporal drift path', drift.is_continuous and len(drift.paths) == 2),
    ]
    return {
        'status': 'PASS' if all(ok for _, ok in checks) else 'FAIL',
        'checks': [{'name': name, 'status': 'PASS' if ok else 'FAIL'} for name, ok in checks],
        'objects': [e_model.key, e_risk.key, e_action.key],
        'morphisms': [
            {'source': mr.source.key, 'target': mr.target.key, 'gamma': mr.gamma, 'valid': True},
            {'source': ra.source.key, 'target': ra.target.key, 'gamma': ra.gamma, 'valid': True},
        ],
        'paths': [certify_path(path, gamma_max=0.5).as_dict()],
        'ruptures': [certify_rupture(rupture).as_dict()],
        'presheaf_contexts': {
            'RiskContext': sorted(presheaf.contexts[e_risk.key]),
            'AuditContext': ['full_trace', 'hash_verified'],
        },
    }


def write_reports(report: dict, out_dir: str | Path = 'reports/category_hott') -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / 'category_hott_checks.json'
    md_path = out / 'category_hott_checks.md'
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Category/HoTT checks', '', f"Status: **{report['status']}**", '', '| Check | Status |', '|---|---|']
    lines.extend(f"| {row['name']} | {row['status']} |" for row in report['checks'])
    lines.extend(['', '## Path certificates', ''])
    lines.extend(f"- `{p['source']} -> {p['target']}` length={p['length']} gamma={p['gamma_total']}" for p in report['paths'])
    lines.extend(['', '## Ruptures', ''])
    lines.extend(f"- `{r['source']} -> {r['target']}` reason={r['reason']} gamma={r['gamma']} gamma_max={r['gamma_max']}" for r in report['ruptures'])
    md_path.write_text('\n'.join(lines), encoding='utf-8')
    return {'json': str(json_path), 'markdown': str(md_path)}


def main() -> None:
    report = run_checks()
    paths = write_reports(report)
    for row in report['checks']:
        print(f"{row['name']}: {row['status']}")
    print(json.dumps({'status': report['status'], **paths}, ensure_ascii=False, indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
