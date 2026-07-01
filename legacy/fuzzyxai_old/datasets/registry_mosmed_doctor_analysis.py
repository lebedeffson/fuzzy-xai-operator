from __future__ import annotations

from pathlib import Path

import pandas as pd

from fuzzyxai.data.dataset_loader import guess_target_column, load_table_dataset
from fuzzyxai.data.dataset_record import DatasetRecord

REGISTRY_CARD_URL = 'https://registry.cit.gov.ru/datasets/c444e643-91be-423f-99bd-abcb63b8f410'
DEFAULT_CANDIDATE_PATHS = [
    Path('data/cit_registry/registry_mosmed_doctor_analysis.csv'),
    Path('data/cit_registry/mosmed_doctor_analysis.csv'),
    Path('data/cit_registry/mosmed_radiography_doctor_results.csv'),
]


def _resolve_path() -> Path:
    for path in DEFAULT_CANDIDATE_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        'Registry dataset "mosmed_doctor_analysis" is not downloaded yet. '
        + 'Put a prepared file in one of: '
        + ', '.join(str(p) for p in DEFAULT_CANDIDATE_PATHS)
        + f'. Source card: {REGISTRY_CARD_URL}'
    )


def _ensure_target(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    target = guess_target_column(df)
    if target is None:
        raise ValueError(
            'Cannot infer target column for registry_mosmed_doctor_analysis. '
            'Add a binary target column (e.g., risk_target) as the last column.'
        )
    return df, str(target)


def load_registry_mosmed_doctor_analysis() -> tuple[DatasetRecord, pd.DataFrame]:
    path = _resolve_path()
    df = load_table_dataset(path)
    df, target = _ensure_target(df)
    record = DatasetRecord(
        name='registry_mosmed_doctor_analysis',
        source='registry.cit.gov.ru',
        local_path=path,
        file_format=path.suffix.lstrip('.').lower(),
        target_column=target,
        task_type='binary_classification',
        description='MosMed doctor analysis results (small medical audit artifact), prepared local slice.',
        metadata={
            'registry_card_url': REGISTRY_CARD_URL,
            'registry_size_mb': 0.62,
            'domain': 'medical_audit',
            'task': 'doctor_model_decision_audit',
            'role': 'медицинский аудит и проверка контекстов применения',
            'dataset_mode': 'registry_mosmed_doctor_analysis',
        },
    )
    return record, df
