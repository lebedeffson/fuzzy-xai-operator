from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORTED_SUFFIXES = {'.csv', '.xlsx', '.xls', '.json', '.parquet'}


def infer_file_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f'Unsupported dataset format: {suffix}')
    return suffix.removeprefix('.')


def load_table_dataset(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    fmt = infer_file_format(path)
    if fmt == 'csv':
        return pd.read_csv(path, **kwargs)
    if fmt in {'xlsx', 'xls'}:
        return pd.read_excel(path, **kwargs)
    if fmt == 'json':
        return pd.read_json(path, **kwargs)
    if fmt == 'parquet':
        return pd.read_parquet(path, **kwargs)
    raise ValueError(f'Unsupported format: {fmt}')


def split_features_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    return df.drop(columns=[target_column]), df[target_column]


def guess_target_column(df: pd.DataFrame) -> str | None:
    if df.empty or len(df.columns) == 0:
        return None
    candidates = ['target', 'label', 'class', 'y', 'result', 'diagnosis', 'risk', 'is_risk']
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name in lower_map:
            return lower_map[name]
    last = df.columns[-1]
    unique = df[last].nunique(dropna=True)
    if unique <= max(20, int(0.05 * max(len(df), 1))):
        return str(last)
    return None
