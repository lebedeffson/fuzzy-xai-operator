from __future__ import annotations

import pandas as pd
import pytest

from fuzzyxai.data import DatasetRecord, infer_file_format
from fuzzyxai.pipelines import DatasetObserverPipeline, write_dataset_observer_report


def _df():
    return pd.DataFrame({
        'age': [30, 44, 52, 61, 73, 39, 48, 57, 69, 35, 42, 66],
        'pressure': [110, 126, 140, 155, 168, 118, 133, 149, 172, 121, 136, 160],
        'marker': [0.10, 0.20, 0.45, 0.71, 0.90, 0.18, 0.38, 0.63, 0.88, 0.22, 0.41, 0.80],
        'target': [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    })


def test_breast_cancer_like_csv_route_contains_all_stages(tmp_path):
    record = DatasetRecord('toy', 'test', target_column='target')
    result = DatasetObserverPipeline().run(record, _df())
    paths = write_dataset_observer_report(result, tmp_path)
    assert result.trace['route'] == 'dataset -> profile -> model -> E_M_ext -> I_pre -> rho -> action'
    assert result.observer_result['action'] in {'accept', 'lower_confidence', 'request_more_data', 'defer_to_human', 'block'}
    assert 'Dataset Observer' in (tmp_path / 'dataset_observer_report.html').read_text()
    assert paths['markdown'].endswith('.md')


def test_csv_without_target_gets_clear_error():
    df = pd.DataFrame({'a': range(30), 'b': range(100, 130)})
    record = DatasetRecord('bad', 'test')
    with pytest.raises(ValueError, match='Target column'):
        DatasetObserverPipeline().run(record, df)


def test_invalid_file_format_gets_clear_error(tmp_path):
    path = tmp_path / 'bad.txt'
    path.write_text('x')
    with pytest.raises(ValueError, match='Unsupported dataset format'):
        infer_file_format(path)
