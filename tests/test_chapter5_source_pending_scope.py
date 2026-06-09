from __future__ import annotations

import csv

from scripts.build_dissertation_artifacts import run


def test_source_pending_scenarios_are_not_run(tmp_path) -> None:
    run(tmp_path / 'artifacts')
    with (tmp_path / 'artifacts/chapter5/table_5_scenario_run_summary.csv').open(encoding='utf-8') as f:
        rows = {row['registry_id']: row for row in csv.DictReader(f)}
    assert rows['gd_anfis_shap']['adapter_called'] == 'True'
    assert rows['gd_anfis_shap']['status'] == 'source-pending'
    for rid in ['beacon_xai', 'gis_integro']:
        assert rows[rid]['adapter_called'] == 'False'
        assert rows[rid]['action'] == 'not_run'
        assert rows[rid]['status'] == 'source-pending'
