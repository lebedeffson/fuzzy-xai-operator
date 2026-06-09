from __future__ import annotations

import csv

from scripts.build_dissertation_artifacts import run


def test_scenarios_keep_claim_scope_without_fake_quantitative_claims(tmp_path) -> None:
    run(tmp_path / 'artifacts')
    with (tmp_path / 'artifacts/chapter5/table_5_scenario_run_summary.csv').open(encoding='utf-8') as f:
        rows = {row['registry_id']: row for row in csv.DictReader(f)}
    assert rows['gd_anfis_shap']['adapter_called'] == 'True'
    assert rows['gd_anfis_shap']['status'] == 'source-pending'
    assert rows['beacon_xai']['adapter_called'] == 'True'
    assert rows['beacon_xai']['status'] == 'fixture-certified'
    assert rows['gis_integro']['adapter_called'] == 'True'
    assert rows['gis_integro']['status'] == 'source-pending'
    for rid in ['gd_anfis_shap', 'beacon_xai', 'gis_integro']:
        assert rows[rid]['action'] == 'audit_report'
        assert 'quantitative' in rows[rid]['claim_scope'] or 'source' in rows[rid]['claim_scope']

    with (tmp_path / 'artifacts/chapter5/table_5_scenario_baseline_comparison.csv').open(encoding='utf-8') as f:
        baseline = {row['registry_id']: row for row in csv.DictReader(f)}
    for rid in ['gd_anfis_shap', 'beacon_xai', 'gis_integro']:
        assert baseline[rid]['quantitative_comparison_available'] == 'false'
        assert baseline[rid]['pinned_baseline_required'] == 'true'
