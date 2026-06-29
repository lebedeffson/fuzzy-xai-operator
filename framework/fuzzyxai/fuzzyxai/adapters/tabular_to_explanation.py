from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from fuzzyxai.core.explanation_object import ExplanationObject, Rule, Trace
from fuzzyxai.hierarchy.f0 import F0


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class TabularToExplanationAdapter:
    """Convert one tabular row + model risk into ExplanationObject."""

    def __init__(self, activation_threshold: float = 0.05) -> None:
        self.activation_threshold = float(activation_threshold)
        self.numeric_columns: list[str] = []

    def fit(self, df: pd.DataFrame, *, target_column: str | None = None) -> 'TabularToExplanationAdapter':
        cols = [str(c) for c in df.columns]
        if target_column and target_column in cols:
            cols = [c for c in cols if c != target_column]
        self.numeric_columns = [str(c) for c in df[cols].select_dtypes(include='number').columns]
        return self

    def adapt(
        self,
        row: pd.Series,
        *,
        sample_id: str,
        predicted_risk: float,
        model_version: str,
        source: str,
        reduction_loss: float = 0.0,
    ) -> ExplanationObject:
        p = _clip01(predicted_risk)
        low = _clip01(1.0 - p * 1.4)
        med = _clip01(1.0 - abs(p - 0.5) * 4.0)
        high = _clip01(p)
        missing_rate = float(row.isna().mean()) if len(row) else 0.0
        uncertainty = _clip01((1.0 - abs(high - low)) * 0.8 + missing_rate * 0.2)

        trace = Trace(
            id=sample_id,
            version=model_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            checksum=f'{sample_id}:{model_version}:{source}',
        )
        rules = [
            Rule('r_low_risk', {'risk': 'low'}, 'accept'),
            Rule('r_medium_risk', {'risk': 'medium'}, 'lower_confidence'),
            Rule('r_high_risk', {'risk': 'high'}, 'defer_to_human'),
        ]
        return ExplanationObject(
            terms={'low_risk', 'medium_risk', 'high_risk'},
            representation=F0(lambda _x, val=p: val, label='tabular_risk'),
            rules=rules,
            activations={'r_low_risk': low, 'r_medium_risk': med, 'r_high_risk': high},
            uncertainty=uncertainty,
            trace=trace,
            reduction_loss=_clip01(reduction_loss),
            metadata={
                'adapter': 'tabular',
                'activation_threshold': self.activation_threshold,
                'missing_rate': missing_rate,
                'n_numeric': len(self.numeric_columns),
            },
        )

