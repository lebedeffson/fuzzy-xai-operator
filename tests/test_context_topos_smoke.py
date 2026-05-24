from __future__ import annotations

from proofs.category_hott_checks import run_checks


def test_category_hott_report_smoke():
    report = run_checks()
    assert report['status'] == 'PASS'
    assert report['paths'][0]['valid'] is True
    assert report['ruptures'][0]['diagnostic_code'] == 'D_ij'
    assert 'RiskContext' in report['presheaf_contexts']
