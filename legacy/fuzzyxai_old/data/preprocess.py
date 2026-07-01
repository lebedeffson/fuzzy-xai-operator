from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .dataset_loader import guess_target_column


@dataclass(frozen=True)
class PreprocessResult:
    target_column: str
    n_rows: int
    n_features: int
    missing_before: float
    missing_after: float
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    metadata: dict[str, Any]


def validate_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError('Dataset is empty')
    if not np.isfinite(df.select_dtypes(include='number').to_numpy(dtype=float, copy=False)).all():
        raise ValueError('Dataset contains inf/-inf values in numeric columns')


def preprocess_dataset(
    df: pd.DataFrame,
    *,
    target_column: str | None = None,
    test_size: float = 0.25,
    random_state: int = 42,
) -> PreprocessResult:
    validate_dataframe(df)
    target = target_column or guess_target_column(df)
    if target is None or target not in df.columns:
        raise ValueError('Target column is not specified and cannot be inferred')

    missing_before = float(df.isna().mean().mean())
    x = df.drop(columns=[target]).copy()
    y = df[target].copy()

    for col in x.columns:
        if pd.api.types.is_numeric_dtype(x[col]):
            x[col] = x[col].fillna(float(x[col].median()) if x[col].notna().any() else 0.0)
        else:
            x[col] = x[col].fillna('unknown').astype(str)

    x = pd.get_dummies(x, dummy_na=False)
    missing_after = float(x.isna().mean().mean())
    if missing_after > 0:
        x = x.fillna(0.0)
        missing_after = float(x.isna().mean().mean())

    stratify = y if y.nunique() <= 20 and y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return PreprocessResult(
        target_column=str(target),
        n_rows=int(len(df)),
        n_features=int(x.shape[1]),
        missing_before=missing_before,
        missing_after=missing_after,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        metadata={'test_size': test_size, 'random_state': random_state},
    )

