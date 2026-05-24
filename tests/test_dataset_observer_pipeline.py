from __future__ import annotations

import pandas as pd

from fuzzyxai.data import DatasetRecord
from fuzzyxai.pipelines import DatasetObserverPipeline, write_dataset_observer_report


def _toy_df():
    return pd.DataFrame({
        'age': [30, 44, 52, 61, 73, 39, 48, 57, 69, 35, 42, 66],
        'pressure': [110, 126, 140, 155, 168, 118, 133, 149, 172, 121, 136, 160],
        'marker': [0.10, 0.20, 0.45, 0.71, 0.90, 0.18, 0.38, 0.63, 0.88, 0.22, 0.41, 0.80],
        'target': [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    })


def test_dataset_observer_pipeline_runs_and_writes_report(tmp_path):
    record = DatasetRecord(name='toy_medical', source='unit-test', target_column='target')
    result = DatasetObserverPipeline(model_name='random_forest').run(record, _toy_df(), case_index=0)
    assert result.accuracy >= 0.0
    assert result.dataset_profile.n_rows == 12
    assert result.observer_result['action'] in {'accept', 'lower_confidence', 'request_more_data', 'defer_to_human', 'block'}
    assert 'application_risk' in result.observer_result
    paths = write_dataset_observer_report(result, tmp_path)
    assert all((tmp_path / name).exists() for name in ['dataset_observer_report.json', 'dataset_observer_report.md', 'dataset_observer_report.html'])
    assert paths['html'].endswith('.html')
