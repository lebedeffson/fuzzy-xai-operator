from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dataset_loader import guess_target_column, load_table_dataset
from .dataset_record import DatasetRecord

DEFAULT_CANDIDATE_PATHS = [
    Path('data/medical/ruccod/ruccod.csv'),
    Path('data/medical/ruccod/ruccod_lite.csv'),
    Path('data/cit_registry/ruccod.csv'),
]


def _fallback_demo() -> tuple[DatasetRecord, pd.DataFrame]:
    rows = [
        {'note_text': 'Acute sepsis signs, urgent care required', 'icd_code': 'A41.9', 'risk_target': 1},
        {'note_text': 'Stable condition, no signs of acute pathology', 'icd_code': 'Z00.0', 'risk_target': 0},
        {'note_text': 'Possible hemorrhage, monitor closely', 'icd_code': 'I62.9', 'risk_target': 1},
        {'note_text': 'Routine follow-up, negative findings', 'icd_code': 'Z09.0', 'risk_target': 0},
        {'note_text': 'Cancer progression suspected', 'icd_code': 'C80.9', 'risk_target': 1},
        {'note_text': 'Normal exam, patient stable', 'icd_code': 'Z01.8', 'risk_target': 0},
    ]
    df = pd.DataFrame(rows)
    record = DatasetRecord(
        name='ruccod_demo_fallback',
        source='fallback:embedded_text',
        target_column='risk_target',
        task_type='binary_classification',
        description='Fallback text+ICD proxy for RuCCoD integration.',
        metadata={'fallback': True, 'fallback_reason': 'local ruccod file not found'},
    )
    return record, df


def load_ruccod_dataset(*, allow_fallback: bool = True) -> tuple[DatasetRecord, pd.DataFrame]:
    for path in DEFAULT_CANDIDATE_PATHS:
        if path.exists():
            df = load_table_dataset(path)
            target = guess_target_column(df)
            if target is None:
                raise ValueError('RuCCoD target column is not found. Add binary target (risk_target).')
            return DatasetRecord(
                name='ruccod',
                source='public-russian-medical',
                local_path=path,
                file_format=path.suffix.removeprefix('.'),
                target_column=str(target),
                task_type='binary_classification',
                description='Russian clinical coding dataset for ICD-aligned risk control.',
                metadata={'fallback': False},
            ), df
    if allow_fallback:
        return _fallback_demo()
    raise FileNotFoundError(
        'RuCCoD dataset not found. Expected one of: '
        + ', '.join(str(p) for p in DEFAULT_CANDIDATE_PATHS)
    )

