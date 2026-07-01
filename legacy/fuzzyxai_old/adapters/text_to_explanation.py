from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from fuzzyxai.core.explanation_object import ExplanationObject, Rule, Trace
from fuzzyxai.hierarchy.f0 import F0


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class TextToExplanationAdapter:
    """Convert medical text + ICD codes into ExplanationObject."""

    HIGH_TOKENS = ('critical', 'sepsis', 'hemorrhage', 'cancer', 'acute')
    LOW_TOKENS = ('stable', 'negative', 'normal', 'no signs')

    def __init__(self, activation_threshold: float = 0.05) -> None:
        self.activation_threshold = float(activation_threshold)

    def adapt(
        self,
        *,
        sample_id: str,
        text: str,
        icd_codes: Iterable[str] | None,
        predicted_risk: float,
        model_version: str = 'model_v1',
        source: str = 'medical_text',
        reduction_loss: float = 0.0,
    ) -> ExplanationObject:
        p = _clip01(predicted_risk)
        low = _clip01(1.0 - p * 1.4)
        med = _clip01(1.0 - abs(p - 0.5) * 4.0)
        high = _clip01(p)

        txt = (text or '').lower()
        has_high = any(tok in txt for tok in self.HIGH_TOKENS)
        has_low = any(tok in txt for tok in self.LOW_TOKENS)
        conflict = float(has_high and has_low)
        uncertainty = _clip01((1.0 - abs(high - low)) * 0.8 + 0.2 * conflict)

        codes = [str(c).strip() for c in (icd_codes or []) if str(c).strip()]
        trace = Trace(
            id=sample_id,
            version=model_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            checksum=f'{sample_id}:{model_version}:{source}',
        )
        return ExplanationObject(
            terms={'low_risk', 'medium_risk', 'high_risk', 'icd_alignment'},
            representation=F0(lambda _x, val=p: val, label='text_risk'),
            rules=[
                Rule('r_text_low', {'risk': 'low'}, 'accept'),
                Rule('r_text_medium', {'risk': 'medium'}, 'lower_confidence'),
                Rule('r_text_high', {'risk': 'high'}, 'defer_to_human'),
            ],
            activations={'r_text_low': low, 'r_text_medium': med, 'r_text_high': high},
            uncertainty=uncertainty,
            trace=trace,
            reduction_loss=_clip01(reduction_loss),
            metadata={
                'adapter': 'text',
                'activation_threshold': self.activation_threshold,
                'icd_codes': codes,
                'text_len': len(text or ''),
                'conflict': bool(conflict),
            },
        )

