from __future__ import annotations

from pathlib import Path

import pandas as pd

from fuzzyxai.data.dataset_loader import guess_target_column, load_table_dataset
from fuzzyxai.data.dataset_record import DatasetRecord

REGISTRY_CARD_URL = 'https://registry.cit.gov.ru/datasets/3e7061cf-3ece-4510-a115-1fc61c369ebf'
DEFAULT_CANDIDATE_PATHS = [
    Path('data/cit_registry/registry_programs.csv'),
    Path('data/cit_registry/programs_registry.csv'),
    Path('data/cit_registry/registered_programs.csv'),
]


def _resolve_path() -> Path:
    for path in DEFAULT_CANDIDATE_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        'Registry dataset "programs" is not downloaded yet. '\
        'Put a prepared file in one of: '
        + ', '.join(str(p) for p in DEFAULT_CANDIDATE_PATHS)
        + f'. Source card: {REGISTRY_CARD_URL}'
    )


def _ensure_target(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    target = guess_target_column(df)
    if target is None:
        raise ValueError(
            'Cannot infer target column for registry_programs. '
            'Add a binary target column (e.g., risk_target) as the last column.'
        )
    return df, str(target)


def load_registry_programs() -> tuple[DatasetRecord, pd.DataFrame]:
    path = _resolve_path()
    df = load_table_dataset(path)
    df, target = _ensure_target(df)
    record = DatasetRecord(
        name='registry_programs',
        source='registry.cit.gov.ru',
        local_path=path,
        file_format=path.suffix.lstrip('.').lower(),
        target_column=target,
        task_type='binary_classification',
        description='Registered software programs dataset (tabular+text), prepared local slice.',
        metadata={
            'registry_card_url': REGISTRY_CARD_URL,
            'registry_size_mb': 6,
            'domain': 'education_science_industry',
            'task': 'text_tabular_ranking',
            'role': 'таблично-текстовый сценарий для объяснения и ранжирования',
            'dataset_mode': 'registry_programs',
        },
    )
    return record, df
