from __future__ import annotations

import pandas as pd
import pytest

from fuzzyxai.data import (
    CITRegistryDatasetClient,
    guess_target_column,
    infer_dataset_profile,
    infer_file_format,
    load_table_dataset,
    split_features_target,
)


def test_table_loader_target_guess_and_split(tmp_path):
    path = tmp_path / 'toy.csv'
    pd.DataFrame({'age': [10, 20, 30], 'pressure': [100, 120, 130], 'target': [0, 1, 0]}).to_csv(path, index=False)
    assert infer_file_format(path) == 'csv'
    df = load_table_dataset(path)
    assert guess_target_column(df) == 'target'
    x, y = split_features_target(df, 'target')
    assert list(x.columns) == ['age', 'pressure']
    assert y.tolist() == [0, 1, 0]


def test_profile_inference_detects_intervals_experts_sources():
    df = pd.DataFrame({
        'risk_min': [0.1, 0.2],
        'risk_max': [0.3, 0.5],
        'expert_a': [1, 0],
        'expert_b': [1, 1],
        'source_ct': [0.8, 0.4],
        'source_lab': [0.7, 0.6],
        'label': [0, 1],
    })
    profile = infer_dataset_profile(df)
    assert profile.has_intervals is True
    assert profile.has_multiple_experts is True
    assert profile.has_source_conflict is True
    assert {'u_int', 'u_expert', 'u_conf'}.issubset(set(profile.suggested_uncertainty_types))


def test_cit_client_wraps_local_file(tmp_path):
    path = tmp_path / 'registry_table.csv'
    pd.DataFrame({'x': [1, 2], 'label': [0, 1]}).to_csv(path, index=False)
    record = CITRegistryDatasetClient(cache_dir=tmp_path / 'cache').from_local_file(path, target_column='label')
    assert record.source == 'registry.cit.gov.ru'
    assert record.file_format == 'csv'
    assert record.as_trace()['target_column'] == 'label'


def test_loader_drops_trailing_empty_unnamed_column(tmp_path):
    path = tmp_path / 'wdbc_like.csv'
    path.write_text('id,diagnosis,x,\n1,M,0.2,\n2,B,0.1,\n', encoding='utf-8')
    df = load_table_dataset(path)
    assert list(df.columns) == ['id', 'diagnosis', 'x']
    assert guess_target_column(df) == 'diagnosis'


def test_cit_client_normalizes_github_blob_url():
    url = 'https://github.com/org/repo/blob/main/data/table.csv'
    assert CITRegistryDatasetClient.normalize_direct_url(url) == 'https://raw.githubusercontent.com/org/repo/main/data/table.csv'


def test_unsupported_dataset_format_is_rejected(tmp_path):
    path = tmp_path / 'bad.txt'
    path.write_text('x')
    with pytest.raises(ValueError):
        infer_file_format(path)
