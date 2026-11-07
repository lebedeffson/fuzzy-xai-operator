"""Security and product-level checks; these tests do not require a model dataset."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from fuzzyxai.doctoral_batch import certify_batch
from fuzzyxai.doctoral_certification import certify_payload, verify_certificate
from fuzzyxai.doctoral_report import render_doctoral_report
from fuzzyxai.doctoral_cli import main
from examples.doctoral_certification.make_control_example import build

TAX = {"method_id": "synthetic_rule_control", "source": "intrinsic", "object": "logic", "audience": "auditor", "canonical_configuration": "synthetic rule"}
REP = {"profile": ["u_num", "u_ling", "u_int"], "selected_class": "F_int", "measured_delta": 0.05}
POL = {"allow_controlled_fixture": True}


def certificate(payload=None):
    return certify_payload(payload if payload is not None else build(), taxonomy=TAX, representation=REP, policy=POL)


def test_html_is_offline_self_contained_and_xss_escaped(tmp_path):
    payload = build()
    payload["claims"][0]["statement"] = '<script>alert("xss")</script>'
    payload["trace"]["dataset_version"] = '<img src=x onerror=alert(1)>'
    payload["explanation_graph"]["nodes"][0]["payload"]["dataset_version"] = payload["trace"]["dataset_version"]
    receipt = certificate(payload)
    assert receipt['decision']['status'] == 'valid'
    page = render_doctoral_report(receipt, tmp_path / "report.html")
    assert '<script>alert' not in page
    assert '&lt;img src=x onerror=alert(1)&gt;' in page
    assert '<!doctype html>' in page
    assert 'https://' not in page
    assert (tmp_path / 'report.html').read_text(encoding='utf-8') == page


def test_html_refuses_tampering(tmp_path):
    result = certificate()
    result['decision']['status'] = 'valid' if result['decision']['status'] != 'valid' else 'invalid'
    with pytest.raises(ValueError):
        render_doctoral_report(result, tmp_path / 'out.html')
    assert not (tmp_path / 'out.html').exists()


def test_batch_invalid_cases_counted_and_no_raw_payloads():
    good = build()
    bad = copy.deepcopy(good)
    bad['risk']['rho'] += 0.2
    report = certify_batch([('b',bad),('a',good),('c',{'risk':{'rho':float('nan')}})],
                           taxonomy=TAX, representation=REP, policy=POL)
    assert report['n_cases'] == 3
    assert report['status_counts']['valid'] == 1
    assert report['status_counts']['invalid'] == 2
    assert report['issue_case_counts']['D_risk_arithmetic'] == 1
    assert report['issue_case_counts']['D_unprocessable_case'] == 1
    assert [x['case_id'] for x in report['cases']] == ['a','b','c']
    assert 'input_sha256' not in json.dumps(report)


def test_batch_rejects_duplicate_names():
    with pytest.raises(ValueError, match='duplicate'):
        certify_batch([('x', build()), ('x', build())], taxonomy=TAX, representation=REP, policy=POL)


def test_canonical_method_wrong_title_is_invalid():
    from fuzzyxai.doctoral_catalog import catalog_lookup
    row = catalog_lookup('m27')
    t = {'method_id':'m27', 'source':row['source'],'object':row['object'],'audience':row['audience'],
         'canonical_configuration':'I invented another method'}
    assert any(v['code'] == 'D_taxonomy' for v in certify_payload(build(),taxonomy=t,representation=REP,policy=POL)['decision']['issues'])


def test_unknown_and_duplicate_claims_fail_closed():
    bad = build();bad['claims'][0]['evidence_status'] = 'verified_by_llm'
    assert any(v['code']=='D_claim_status' for v in certificate(bad)['decision']['issues'])
    dup = build();dup['claims'].append(copy.deepcopy(dup['claims'][0]))
    assert any(v['code']=='D_claim_identity' for v in certificate(dup)['decision']['issues'])


def test_proof_redacts_sensitive_raw_content_recursively():
    x=build()
    x['layers']['text_highlights'] = [{'raw_text': 'secret tokens'}]
    x['model']={'image_png_base64':'sensitive bytes'}
    receipt = certificate(x)
    assert verify_certificate(receipt)['valid']
    wire=json.dumps(receipt)
    assert 'secret tokens' not in wire
    assert 'sensitive bytes' not in wire
    assert 'raw payload omitted' in wire


def test_cli_html_and_batch_flow(tmp_path):
    src = tmp_path/'inputs'
    src.mkdir()
    (src/'case_1.json').write_text(json.dumps(build()), encoding='utf-8')
    for name,val in [('taxonomy',TAX),('representation',REP),('policy',POL)]:
        (tmp_path/f'{name}.json').write_text(json.dumps(val),encoding='utf-8')
    args = ['certify','--input',str(src/'case_1.json'),'--taxonomy',str(tmp_path/'taxonomy.json'),
            '--representation',str(tmp_path/'representation.json'),'--policy',str(tmp_path/'policy.json'),
            '--out-dir',str(tmp_path/'report'),'--html']
    assert main(args) == 0
    assert (tmp_path/'report'/'report.html').exists()
    assert main(['verify','--receipt',str(tmp_path/'report'/'certificate.json')]) == 0
    assert main(['batch','--input-dir',str(src),'--taxonomy',str(tmp_path/'taxonomy.json'),
                 '--representation',str(tmp_path/'representation.json'),'--policy',str(tmp_path/'policy.json'),
                 '--out-dir',str(tmp_path/'batch')]) == 0
    data=json.loads((tmp_path/'batch'/'batch_summary.json').read_text(encoding='utf-8'))
    assert data['status_counts'] == {'valid': 1}
    assert (tmp_path/'batch'/'batch_cases.csv').exists()


def test_long_graph_fails_without_recursion_error():
    x=build()
    previous='prediction'
    # Keep an alternate valid route but create a long acyclic chain.
    for i in range(1200):
        node=f'intermediate:{i}'
        x['explanation_graph']['nodes'].append({'node_id':node,'node_type':'intermediate'})
        x['explanation_graph']['edges'].append({'source':previous,'target':node})
        previous=node
    assert certificate(x)['decision']['status']=='valid'
    # Introduce a back-edge and catch DAG violation, not recursion depth.
    x['explanation_graph']['edges'].append({'source':previous,'target':'prediction'})
    result=certificate(x)
    assert any(v['code']=='D_graph_cycle' for v in result['decision']['issues'])


def test_representation_must_be_executed_not_only_chosen():
    data=build()
    data['fuzzy'].pop('selected_representation')
    result=certificate(data)
    assert 'D_representation_not_executed' in [v['code'] for v in result['decision']['issues']]
    assert result['decision']['status']=='insufficient_evidence'


def test_representation_execution_mismatch_caught():
    data=build()
    data['fuzzy']['selected_representation']='F0'
    result=certificate(data)
    assert 'D_representation_mismatch' in [v['code'] for v in result['decision']['issues']]


def test_profile_cannot_hide_structural_rupture():
    data=build()
    data['risk']['components']['chi_R'] = 1
    data['risk']['rho'] += data['risk']['risk_weights']['chi_R']
    result=certificate(data)
    assert 'D_profile_omits_conflict' in [v['code'] for v in result['decision']['issues']]


def test_source_conflict_requires_source_channel():
    data=build()
    data['diagnostics'] = [{'diagnostic_id': 'D_quality_source_conflict'}]
    rep={**REP,'profile':['u_num','u_ling','u_conf','u_trace'],'selected_class':'FML_audit'}
    result=certify_payload(data,taxonomy=TAX,representation=rep,policy=POL)
    assert 'D_profile_omits_source' in [v['code'] for v in result['decision']['issues']]


def test_membership_degree_replayed_from_original_value():
    original = build()
    assert certificate(original)['decision']['status'] == 'valid'
    altered = copy.deepcopy(original)
    altered['layers']['fuzzy_rule_activations'][0]['terms'][0]['feature_value'] = 0.7
    result = certificate(altered)
    assert 'D_membership_replay' in [v['code'] for v in result['decision']['issues']]


def test_membership_policy_and_input_absent_are_not_fabricated():
    no_policy = build()
    no_policy['fuzzy']['membership_policies'].pop('context')
    assert 'D_term_policy_missing' in [v['code'] for v in certificate(no_policy)['decision']['issues']]
    no_value = build()
    no_value['layers']['fuzzy_rule_activations'][0]['terms'][0].pop('feature_value')
    assert 'D_term_input_missing' in [v['code'] for v in certificate(no_value)['decision']['issues']]


def test_receipt_binds_exact_certifier_code_version():
    valid = certificate()
    assert valid['certifier_code_sha256']
    assert verify_certificate(valid)['certifier_code_matches']
    forged = copy.deepcopy(valid)
    forged['certifier_code_sha256'] = '0' * 64
    assert not verify_certificate(forged)['valid']


def test_cross_operator_preinterpretability_is_not_free_number():
    data=build()
    data['disagreement']['pre_interpretability']=.4
    result=certificate(data)
    assert 'D_i_pre_risk_mismatch' in [v['code'] for v in result['decision']['issues']]


def test_delta_used_by_risk_must_be_executed_delta():
    data=build()
    data['risk']['components']['Delta'] = 0.12
    data['risk']['rho'] += (0.12-0.05) * data['risk']['risk_weights']['Delta']
    result=certificate(data)
    assert 'D_risk_reduction_mismatch' in [v['code'] for v in result['decision']['issues']]


def test_proof_action_must_match_risk_action():
    data=build()
    data['explanation_graph']['nodes'][-1]['payload']['action'] = 'block'
    result=certificate(data)
    assert 'D_graph_action_mismatch' in [v['code'] for v in result['decision']['issues']]


def test_missing_i_pre_and_action_provenance_fail_closed():
    data=build()
    del data['disagreement']['pre_interpretability']
    del data['explanation_graph']['nodes'][-1]['payload']
    result=certificate(data)
    codes={v['code'] for v in result['decision']['issues']}
    assert {'D_i_pre_missing', 'D_graph_action_missing'} <= codes
    assert result['decision']['status']=='insufficient_evidence'


def test_physical_domain_membership_functions_not_forced_into_zero_one():
    from fuzzyxai.doctoral_certification import triangular_membership
    assert triangular_membership(75.0, [70.0, 80.0, 90.0]) == .5
    assert triangular_membership(-10.0, [-20.0, -10.0, 0.0]) == 1.0
    with pytest.raises(ValueError):
        triangular_membership('75', [70, 80, 90])
