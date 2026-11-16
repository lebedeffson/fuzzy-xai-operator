"""Reproducible controlled intervention experiment for TISU.

Source: exact upstream core files at commit aae7aa717bff913240cbb0026807cd1ccfa67160.
Uses actual held-out probabilities from scikit-learn Breast Cancer Wisconsin.
Each error is injected explicitly; this is contract-conformance, NOT an estimate
of generalization to unknown faults or of clinical safety.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from fuzzyxai.core.composition import compose
from fuzzyxai.core.composition_checked import (
    CompositionPolicy, OriginRecord, MembershipSpec, compose_checked, result_status,
)
from fuzzyxai.core.explanation_object import ExplanationObject, Rule, Trace
from fuzzyxai.hierarchy.f0 import F0

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('tisu_validation_outputs')
BETA = {
    'repr': 0.20, 'rules': 0.20, 'activations': 0.20,
    'uncertainty': 0.15, 'trace': 0.15, 'reduction': 0.10,
    'gamma_max': 0.25,
}
POLICY = CompositionPolicy(
    max_score_difference=0.05,
    max_membership_distance=0.08,
    require_origin_record=True,
    verify_membership=True,
)
TYPED_ONLY = replace(POLICY, verify_membership=False)
MODEL_ID = 'sklearn_random_forest_bcw'
MODEL_VERSION = 'rf_v1_seed42'
TSTAMP = '2026-11-16T12:00:00Z'
SCENARIOS = (
    ('valid_identity', 'ACCEPT'),
    ('valid_complement', 'ACCEPT'),
    ('valid_score_noise', 'ACCEPT'),
    ('valid_membership_noise', 'ACCEPT'),
    ('wrong_event_no_mapping', 'REJECT'),
    ('wrong_mapping', 'REJECT'),
    ('conflicting_rule_signature', 'REJECT'),
    ('model_version_mismatch', 'REJECT'),
    ('sample_id_mismatch', 'REJECT'),
    ('missing_trace_source', 'UNKNOWN'),
    ('missing_trace_checksum', 'UNKNOWN'),
    ('missing_target_event', 'UNKNOWN'),
    ('missing_independent_origin', 'UNKNOWN'),
    ('disjoint_vocabulary', 'REJECT'),
    ('large_membership_shift', 'REJECT'),
    ('grid_alias_membership', 'REJECT'),
    ('missing_membership_definition', 'UNKNOWN'),
    ('score_inconsistency', 'REJECT'),
    ('invalid_probability', 'REJECT'),
    ('coordinated_wrong_event', 'REJECT'),
)


def term_specs(high_shift: float = 0.0, *, grid_alias: bool = False) -> dict[str, MembershipSpec]:
    if grid_alias:
        # Identical at every upstream F0 grid point k/100, but a narrow
        # break between 0.00 and 0.01 changes the function by 0.6.
        high = ((0.0, 0.0), (0.004, 0.0), (0.005, 0.6), (0.006, 0.0),
                (0.5, 0.0), (0.75, 1.0), (1.0, 1.0))
    else:
        high = ((0.0, 0.0), (0.5 + high_shift, 0.0),
                (0.75 + high_shift, 1.0), (1.0, 1.0))
    return {
        'low': MembershipSpec(((0.0, 1.0), (0.5, 0.0), (1.0, 0.0))),
        'medium': MembershipSpec(((0.0, 0.0), (0.25, 0.0), (0.5, 1.0),
                                  (0.75, 0.0), (1.0, 0.0))),
        'high': MembershipSpec(high),
    }


def as_f0(high: MembershipSpec) -> F0:
    return F0(high.value, 'high-risk')


def serialize_specs(specs: dict[str, MembershipSpec]) -> dict[str, list[list[float]]]:
    return {name: obj.to_json() for name, obj in specs.items()}


def trace_checksum(sample_id: str, probability: float, module_name: str,
                   model_id: str = MODEL_ID, model_version: str = MODEL_VERSION) -> str:
    """SHA-256 of the declared output tuple; not an authenticated signature."""
    payload = json.dumps({'sample_id': sample_id, 'model_id': model_id, 'model_version': model_version,
                          'p_malignant': float(probability), 'module': module_name},
                         sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def build_pair(dataset_index: int, p_malignant: float, *,
               model_id: str = MODEL_ID, model_version: str = MODEL_VERSION):
    name = f'bcw-{dataset_index:03d}'
    origin = OriginRecord(name, model_id, model_version, 'malignant', float(p_malignant))
    # Margin uncertainty is a heuristic uncertainty score, not a calibrated probability.
    u = float(1.0 - abs(2.0 * p_malignant - 1.0))
    specs = term_specs()
    a = ExplanationObject(
        terms={'low', 'medium', 'high'},
        representation=as_f0(specs['high']),
        rules=[Rule('risk_rule', {'risk': 'high'}, 'expert_review')],
        activations={'risk_rule': specs['high'].value(float(p_malignant))},
        uncertainty=u,
        trace=Trace(f'{name}:model', 'module-v1', TSTAMP,
                    source='random_forest', checksum=trace_checksum(name, p_malignant, 'model', model_id, model_version)),
        metadata={
            'sample_id': name, 'model_id': model_id,
            'model_version': model_version, 'target_event': 'malignant',
            'probability': float(p_malignant), 'scale': 'probability',
            'mapping': 'identity', 'term_specs': serialize_specs(specs),
            'fuzzy_event': 'malignant', 'representation_term': 'high',
        },
    )
    b = a.copy_with(
        trace=Trace(f'{name}:decision', 'module-v1', TSTAMP,
                    source='decision_rule', checksum=trace_checksum(name, p_malignant, 'decision', model_id, model_version)),
        uncertainty=min(1.0, u + 0.02),
    )
    return a, b, origin


def inject(name: str, a: ExplanationObject, b: ExplanationObject, origin: OriginRecord):
    # All transformations are defined before running any method, not selected post hoc.
    if name == 'valid_identity':
        pass
    elif name == 'valid_complement':
        b.metadata['target_event'] = 'benign'
        b.metadata['probability'] = 1.0 - float(origin.probability)
        b.metadata['mapping'] = 'complement'
    elif name == 'valid_score_noise':
        p = float(origin.probability)
        b.metadata['probability'] = p + (0.015 if p <= 0.85 else -0.015)
    elif name == 'valid_membership_noise':
        ts = term_specs(high_shift=0.01)
        b.metadata['term_specs'] = serialize_specs(ts)
        b.representation = as_f0(ts['high'])
        b.activations['risk_rule'] = ts['high'].value(float(origin.probability))
    elif name == 'wrong_event_no_mapping':
        b.metadata['target_event'] = 'benign'
        b.metadata['probability'] = float(origin.probability)
    elif name == 'wrong_mapping':
        b.metadata['target_event'] = 'benign'
        b.metadata['probability'] = 1.0 - float(origin.probability)
        b.metadata['mapping'] = 'identity'
    elif name == 'conflicting_rule_signature':
        b.rules = [Rule('risk_rule', {'risk': 'high'}, 'no_review')]
    elif name == 'model_version_mismatch':
        b.metadata['model_version'] = 'rf_v2_unapproved'
    elif name == 'sample_id_mismatch':
        b.metadata['sample_id'] = 'bcw-other'
    elif name == 'missing_trace_source':
        b.trace = replace(b.trace, source='')
    elif name == 'missing_trace_checksum':
        b.trace = replace(b.trace, checksum='')
    elif name == 'missing_target_event':
        del b.metadata['target_event']
    elif name == 'missing_independent_origin':
        origin = None
    elif name == 'disjoint_vocabulary':
        b.terms = {'unrelated'}
    elif name == 'large_membership_shift':
        ts = term_specs(high_shift=0.12)
        b.metadata['term_specs'] = serialize_specs(ts)
        b.representation = as_f0(ts['high'])
        b.activations['risk_rule'] = ts['high'].value(float(origin.probability))
    elif name == 'grid_alias_membership':
        ts = term_specs(grid_alias=True)
        b.metadata['term_specs'] = serialize_specs(ts)
        b.representation = as_f0(ts['high'])
    elif name == 'missing_membership_definition':
        del b.metadata['term_specs']
    elif name == 'score_inconsistency':
        p = float(origin.probability)
        b.metadata['probability'] = p + (0.13 if p < 0.7 else -0.13)
    elif name == 'invalid_probability':
        b.metadata['probability'] = 1.2
    elif name == 'coordinated_wrong_event':
        a.metadata['target_event'] = 'benign'
        b.metadata['target_event'] = 'benign'
        a.metadata['fuzzy_event'] = 'benign'
        b.metadata['fuzzy_event'] = 'benign'
        # Both modules keep p(malignant) but falsely label it p(benign).
        # The only reliable signal is the independently stored origin event.
    else:
        raise AssertionError(f'unknown scenario {name}')
    return a, b, origin


def simple_numeric(a: ExplanationObject, b: ExplanationObject) -> str:
    return 'ACCEPT' if abs(float(a.metadata['probability']) - float(b.metadata['probability'])) <= 0.05 else 'REJECT'


def upstream(a: ExplanationObject, b: ExplanationObject) -> str:
    return result_status(compose(a, b, BETA))


def run(*, out_dir: Path | None = None) -> dict:
    global OUT
    if out_dir is not None:
        OUT = Path(out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = load_breast_cancer(as_frame=True)
    x = dataset.data
    y = (dataset.target == 0).astype(int)  # 1 = malignant
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y,
    )
    model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42, n_jobs=1)
    model.fit(x_train, y_train)
    index_malignant = list(model.classes_).index(1)
    ps = model.predict_proba(x_test)[:, index_malignant]
    preds = (ps >= 0.5).astype(int)
    model_metrics = {
        'n_total': int(len(x)), 'n_train': int(len(x_train)), 'n_test': int(len(x_test)),
        'n_features': int(x.shape[1]), 'n_estimators': 120, 'max_depth': 6,
        'accuracy': round(float(accuracy_score(y_test, preds)), 6),
        'roc_auc': round(float(roc_auc_score(y_test, ps)), 6),
        'dataset': 'scikit-learn load_breast_cancer (Wisconsin Diagnostic Breast Cancer)',
        'malignant_label_after_conversion': 1,
    }
    by_type = defaultdict(lambda: {m: Counter() for m in ('numeric', 'upstream', 'typed_only', 'checked')})
    records = []
    for dataset_index, p in zip(x_test.index, ps):
        for scenario, expected in SCENARIOS:
            a, b, anchor = build_pair(int(dataset_index), float(p))
            a, b, anchor = inject(scenario, a, b, anchor)
            labels = {
                'numeric': simple_numeric(a, b),
                'upstream': upstream(a, b),
                'typed_only': result_status(compose_checked(a, b, BETA, origin=anchor, policy=TYPED_ONLY)),
                'checked': result_status(compose_checked(a, b, BETA, origin=anchor, policy=POLICY)),
            }
            for m, status in labels.items():
                by_type[scenario][m][status] += 1
            records.append({'dataset_index': int(dataset_index), 'p_malignant': round(float(p), 9),
                            'scenario': scenario, 'expected': expected, **labels})
    n = len(records)
    methods = ['numeric', 'upstream', 'typed_only', 'checked']
    totals = {}
    for m in methods:
        correct = sum(r[m] == r['expected'] for r in records)
        false_accept = sum(r[m] == 'ACCEPT' and r['expected'] == 'REJECT' for r in records)
        missing_accept = sum(r[m] == 'ACCEPT' and r['expected'] == 'UNKNOWN' for r in records)
        false_reject = sum(r[m] == 'REJECT' and r['expected'] == 'ACCEPT' for r in records)
        totals[m] = {'correct_status': correct, 'total': n, 'exact_match_pct': round(100*correct/n, 3),
                     'false_accept_incompatible': false_accept, 'false_accept_incomplete': missing_accept,
                     'false_reject_valid': false_reject}
    scenario_rows=[]
    for name, expected in SCENARIOS:
        item={'scenario':name,'expected':expected,'n':int(len(x_test))}
        for m in methods:
            counts=by_type[name][m]
            item[m]=max(('ACCEPT','REJECT','UNKNOWN'),key=lambda s:counts[s])
            item[f'{m}_counts']={s:int(counts[s]) for s in ('ACCEPT','REJECT','UNKNOWN')}
        scenario_rows.append(item)
    example_idx=int(np.argmin(np.abs(ps-0.70)))
    p_ex=float(ps[example_idx])
    a,b,anchor=build_pair(int(x_test.index[example_idx]),p_ex)
    res=compose_checked(a,b,BETA,origin=anchor,policy=POLICY)
    assert result_status(res)=='ACCEPT'
    canonical= {
        'sample_id':anchor.sample_id,'p_malignant':round(p_ex,9),'p_benign':round(1-p_ex,9),
        'mu_high':round(float(a.representation.membership(p_ex)),9),
        'margin_uncertainty':round(float(a.uncertainty),9),
        'gamma':round(float(res.metadata['gamma']),9),
        'composed_uncertainty':round(float(res.uncertainty),9),
    }
    # Correctness checks (pre-registered targets by intervention, not by detector output).
    for record in records:
        if record['checked'] != record['expected']:
            raise AssertionError(f"Contract check failed: {record}")
    # The source function is unchanged and must demonstrably miss at least one injected inconsistency.
    assert totals['upstream']['false_accept_incompatible'] > 0
    # A basic type-only ablation intentionally has no membership-function check.
    assert any(r['typed_only']=='ACCEPT' for r in records if r['scenario']=='large_membership_shift')
    assert all(r['checked']=='REJECT' for r in records if r['scenario']=='large_membership_shift')
    # Removal of independent origin makes a coordinated semantic change undetectable pairwise.
    a,b,anchor=build_pair(int(x_test.index[example_idx]),p_ex)
    a,b,anchor=inject('coordinated_wrong_event',a,b,anchor)
    no_anchor_policy=replace(POLICY,require_origin_record=False)
    pairwise_status=result_status(compose_checked(a,b,BETA,origin=None,policy=no_anchor_policy))
    assert pairwise_status=='ACCEPT'
    config={
        'split_seed':42,'test_fraction':0.25,'stratify':True,
        'test_observations':int(len(x_test)),'intervention_types':len(SCENARIOS),
        'intervention_pairs':n,
        'score_threshold':POLICY.max_score_difference,
        'membership_threshold':POLICY.max_membership_distance,
        'membership_metric':'exact sup norm on union of knots of piecewise-linear functions; original F0 uses 101-point grid',
        'beta':BETA, 'domain':'binary malignant/benign probability; F0 on canonical malignant risk',
        'anchor':'independent pre-injection model output stored by evaluator (not a digital signature)',
    }
    core_paths = ('core/composition.py', 'core/composition_checked.py',
                  'core/trust_evaluator.py', 'hierarchy/f0.py')
    package_root = ROOT / 'framework' / 'fuzzyxai' / 'fuzzyxai'
    hashes = {rel: hashlib.sha256((package_root / rel).read_bytes()).hexdigest()
              for rel in core_paths}
    report={
        'source_revision':'aae7aa717bff913240cbb0026807cd1ccfa67160',
        'scope':'component-level: genuine sklearn test predictions + unchanged upstream compose() and F0; not whole framework deployment',
        'python':platform.python_version(),'numpy':np.__version__,'scikit_learn':sklearn.__version__,
        'model':model_metrics,'config':config,'example':canonical,
        'totals':totals,'scenarios':scenario_rows,
        'pairwise_without_anchor_coordinated_event':pairwise_status,
        'sha256_source_files':hashes,
        'note':'Counts are controlled scenario-conformance checks on repeated held-out predictions, not independent examples of naturally occurring failures; thresholds fixed before execution.',
    }
    (OUT/'experiment_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'all_cases.csv').open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (OUT/'scenario_summary.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['scenario','expected','n','numeric','upstream','typed_only','checked']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in scenario_rows:
            writer.writerow({k: row[k] for k in fields})
    print(json.dumps({'model':model_metrics,'example':canonical,'totals':totals,
                      'pairwise_without_anchor':pairwise_status,'n_scenarios':len(SCENARIOS)},
                     ensure_ascii=False,indent=2))
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Controlled TISU F0 composition benchmark')
    parser.add_argument('--out-dir', type=Path, default=Path('tisu_validation_outputs'))
    args = parser.parse_args()
    run(out_dir=args.out_dir)
