from __future__ import annotations

import csv
import json
from pathlib import Path

from fuzzyxai.experiments.chapter3_critical_ruptures import N_OBJECTS, POLICIES, run


def test_chapter3_critical_ruptures_outputs_all_required_artifacts() -> None:
    summary = run()
    assert summary['status'] == 'ok'
    assert summary['n_objects'] == N_OBJECTS
    assert summary['seed']
    assert summary['input_checksum_sha256']
    assert summary['results_checksum_sha256']
    assert summary['command'] == 'make reproduce-critical-ruptures'

    root = Path.cwd()
    summary_path = root / str(summary['results_csv']).replace('results.csv', 'summary.json')
    results_path = root / summary['results_csv']
    figure_path = root / summary['figure']
    manifest_path = root / 'evidence' / 'critical_ruptures_manifest_sha256.json'

    assert summary_path.exists()
    assert results_path.exists()
    assert figure_path.exists()
    assert manifest_path.exists()

    with results_path.open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert {row['policy'] for row in rows} == set(POLICIES)
    assert len(rows) == 5
    required = {
        'n_objects',
        'critical_objects',
        'missed_critical_ruptures',
        'critical_rupture_recall',
        'false_block_rate',
        'agreement_reference',
        'mean_processing_time_ms',
    }
    for row in rows:
        assert required <= set(row)
        assert int(row['n_objects']) == 1000
        for field in required - {'n_objects'}:
            assert row[field] not in {'', 'None', 'nan'}

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['n_objects'] == 1000
    assert set(manifest['files']) == {
        'reports/chapter3/synthetic_ruptures_summary.json',
        'reports/chapter3/synthetic_ruptures_results.csv',
        'figures/chapter3/critical_ruptures_comparison.png',
    }
