from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetProfile:
    n_rows: int
    n_columns: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    missing_rate: float
    has_intervals: bool
    has_multiple_experts: bool
    has_source_conflict: bool
    requires_audit: bool
    suggested_uncertainty_types: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _has_interval_pairs(columns: list[str]) -> bool:
    names = {c.lower() for c in columns}
    bases = set()
    for c in names:
        for suffix in ('_min', '_max', '_lower', '_upper', '_lo', '_hi'):
            if c.endswith(suffix):
                bases.add(c[: -len(suffix)])
    return any(
        (f'{b}_min' in names and f'{b}_max' in names)
        or (f'{b}_lower' in names and f'{b}_upper' in names)
        or (f'{b}_lo' in names and f'{b}_hi' in names)
        for b in bases
    )


def infer_dataset_profile(
    df: pd.DataFrame,
    *,
    requires_audit: bool = True,
    expert_columns_prefix: str = 'expert_',
    source_columns_prefix: str = 'source_',
) -> DatasetProfile:
    numeric_columns = [str(c) for c in df.select_dtypes(include=[np.number]).columns]
    categorical_columns = [str(c) for c in df.columns if str(c) not in numeric_columns]
    missing_rate = float(df.isna().mean().mean()) if len(df.columns) else 0.0
    columns = [str(c) for c in df.columns]
    has_intervals = _has_interval_pairs(columns)
    expert_cols = [c for c in columns if c.lower().startswith(expert_columns_prefix)]
    source_cols = [c for c in columns if c.lower().startswith(source_columns_prefix)]
    has_multiple_experts = len(expert_cols) >= 2
    has_source_conflict = len(source_cols) >= 2

    uncertainty_types = ['u_num'] if numeric_columns else []
    if categorical_columns:
        uncertainty_types.append('u_ling')
    if missing_rate > 0:
        uncertainty_types.append('u_trace')
    if has_intervals:
        uncertainty_types.append('u_int')
    if has_multiple_experts:
        uncertainty_types.append('u_expert')
    if has_source_conflict:
        uncertainty_types.append('u_conf')
    if requires_audit:
        uncertainty_types.append('u_trace')

    return DatasetProfile(
        n_rows=int(len(df)),
        n_columns=int(len(df.columns)),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        missing_rate=missing_rate,
        has_intervals=has_intervals,
        has_multiple_experts=has_multiple_experts,
        has_source_conflict=has_source_conflict,
        requires_audit=requires_audit,
        suggested_uncertainty_types=sorted(set(uncertainty_types)),
        metadata={'expert_columns': expert_cols, 'source_columns': source_cols},
    )
