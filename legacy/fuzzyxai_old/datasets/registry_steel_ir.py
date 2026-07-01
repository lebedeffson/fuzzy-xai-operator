from __future__ import annotations

from pathlib import Path

import pandas as pd

from fuzzyxai.data.dataset_loader import guess_target_column, load_table_dataset
from fuzzyxai.data.dataset_record import DatasetRecord

REGISTRY_CARD_URL = 'https://registry.cit.gov.ru/datasets/99eb99d2-cad4-4c47-a017-2ae927018478'
DEFAULT_CANDIDATE_PATHS = [
    Path('data/cit_registry/registry_steel_ir.csv'),
    Path('data/cit_registry/steel_ir_features.csv'),
    Path('data/cit_registry/industrial_steel_ir.csv'),
]


def _resolve_path() -> Path:
    for path in DEFAULT_CANDIDATE_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        'Registry dataset "steel_ir" is not downloaded yet. '
        + 'Put a prepared file in one of: '
        + ', '.join(str(p) for p in DEFAULT_CANDIDATE_PATHS)
        + f'. Source card: {REGISTRY_CARD_URL}'
    )


def _ensure_target(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    target = guess_target_column(df)
    if target is None:
        raise ValueError(
            'Cannot infer target column for registry_steel_ir. '
            'Add a binary target column (e.g., risk_target) as the last column.'
        )
    return df, str(target)


def load_registry_steel_ir() -> tuple[DatasetRecord, pd.DataFrame]:
    path = _resolve_path()
    df = load_table_dataset(path)
    df, target = _ensure_target(df)
    record = DatasetRecord(
        name='registry_steel_ir',
        source='registry.cit.gov.ru',
        local_path=path,
        file_format=path.suffix.lstrip('.').lower(),
        target_column=target,
        task_type='binary_classification',
        description='Industrial IR steel inspection dataset, prepared tabular feature slice.',
        metadata={
            'registry_card_url': REGISTRY_CARD_URL,
            'registry_size_mb': 76,
            'domain': 'industrial_cv',
            'task': 'object_detection_tracking',
            'role': 'промышленный контроль и переносимость метода',
            'dataset_mode': 'registry_steel_ir',
        },
    )
    return record, df
