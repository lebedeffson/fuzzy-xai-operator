from __future__ import annotations

import csv
import json

from scripts.build_dissertation_artifacts import run


def test_dissertation_artifacts_pack_contains_chapter_tables(tmp_path) -> None:
    result = run(tmp_path / 'dissertation_artifacts')
    root = tmp_path / 'dissertation_artifacts'
    assert result['status'] == 'ok'
    assert (root / 'chapter2' / 'table_2_sample113_values.csv').exists()
    assert (root / 'chapter4' / 'table_4_evidence_matrix.csv').exists()
    assert (root / 'chapter5' / 'table_5_scenario_run_summary.csv').exists()
    assert (root / 'chapter5' / 'table_5_scenario_quantitative_summary.csv').exists()
    assert (root / 'chapter5' / 'fig_5_module_channel_coverage.png').exists()
    assert (root / 'diagram_specs' / 'chapter5' / 'fig_5_2_hybrid_xiris_route.json').exists()
    assert (root / 'retained_figures_manifest.csv').exists()

    with (root / 'chapter4' / 'table_4_evidence_matrix.csv').open(encoding='utf-8') as f:
        row = next(csv.DictReader(f))
    assert {'module_id', 'evidence_level', 'status', 'source_repo', 'claim_scope'} <= set(row)

    with (root / 'chapter5' / 'table_5_scenario_run_summary.csv').open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        assert len(reader.fieldnames) == len(set(reader.fieldnames))
        scenario_ids = {row['registry_id'] for row in reader}
    assert {'gd_anfis_shap', 'hybrid_xiris', 'anza_lira', 'beacon_xai', 'gis_integro'} <= scenario_ids
    assert (root / 'chapter5' / 'text_5_scenario_run_insert.md').read_text(encoding='utf-8').count('not_available') >= 1

    manifest = json.loads((root / 'artifact_manifest_sha256.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'ok'
    assert manifest['files']
