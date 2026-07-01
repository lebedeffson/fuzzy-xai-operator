from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_diabetes

from .dataset_loader import guess_target_column, load_table_dataset
from .dataset_record import DatasetRecord

DEFAULT_CANDIDATE_PATHS = [
    Path('data/medical/rikord/rikord.csv'),
    Path('data/medical/rikord/rikord_tabular.csv'),
    Path('data/cit_registry/rikord.csv'),
]


def _fallback_demo() -> tuple[DatasetRecord, pd.DataFrame]:
    ds = load_diabetes(as_frame=True)
    df = ds.frame.copy()
    threshold = float(df['target'].median())
    df['risk_target'] = (df['target'] >= threshold).astype(int)
    df = df.drop(columns=['target'])
    record = DatasetRecord(
        name='rikord_demo_fallback',
        source='fallback:sklearn_diabetes',
        target_column='risk_target',
        task_type='binary_classification',
        description='Fallback tabular proxy for RIKORD pipeline integration.',
        metadata={'fallback': True, 'fallback_reason': 'local rikord file not found'},
    )
    return record, df


def load_rikord_dataset(*, allow_fallback: bool = True) -> tuple[DatasetRecord, pd.DataFrame]:
    for path in DEFAULT_CANDIDATE_PATHS:
        if path.exists():
            df = load_table_dataset(path)
            target = guess_target_column(df)
            if target is None:
                raise ValueError('RIKORD target column is not found. Add binary target (risk_target).')
            return DatasetRecord(
                name='rikord',
                source='public-russian-medical',
                local_path=path,
                file_format=path.suffix.removeprefix('.'),
                target_column=str(target),
                task_type='binary_classification',
                description='RIKORD-like ICU tabular dataset for sepsis risk prediction.',
                metadata={'fallback': False},
            ), df
    if allow_fallback:
        return _fallback_demo()
    raise FileNotFoundError(
        'RIKORD dataset not found. Expected one of: '
        + ', '.join(str(p) for p in DEFAULT_CANDIDATE_PATHS)
    )

