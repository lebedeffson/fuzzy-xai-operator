from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from fuzzyxai.core.explanation_object import ExplanationObject, Rule, Trace
from fuzzyxai.hierarchy.f0 import F0


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class MedicalImageToExplanationAdapter:
    """Build ExplanationObject for medical image sample (DICOM/meta compatible)."""

    def __init__(self, activation_threshold: float = 0.05) -> None:
        self.activation_threshold = float(activation_threshold)

    def adapt(
        self,
        *,
        sample_id: str,
        predicted_risk: float,
        image_path: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        model_version: str = 'model_v1',
        source: str = 'medical_image',
        reduction_loss: float = 0.0,
    ) -> ExplanationObject:
        p = _clip01(predicted_risk)
        low = _clip01(1.0 - p * 1.4)
        med = _clip01(1.0 - abs(p - 0.5) * 4.0)
        high = _clip01(p)
        meta = dict(metadata or {})
        if image_path:
            meta['image_path'] = str(image_path)
            meta['image_suffix'] = Path(image_path).suffix.lower()

        modality = str(meta.get('modality', '')).strip()
        protocol = str(meta.get('protocol', '')).strip()
        missing_meta = int(not modality) + int(not protocol)
        uncertainty = _clip01((1.0 - abs(high - low)) * 0.75 + 0.125 * missing_meta)

        trace = Trace(
            id=sample_id,
            version=model_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            checksum=f'{sample_id}:{model_version}:{source}',
        )
        return ExplanationObject(
            terms={'low_risk', 'medium_risk', 'high_risk', 'image_quality'},
            representation=F0(lambda _x, val=p: val, label='medical_image_risk'),
            rules=[
                Rule('r_image_low', {'risk': 'low'}, 'accept'),
                Rule('r_image_medium', {'risk': 'medium'}, 'lower_confidence'),
                Rule('r_image_high', {'risk': 'high'}, 'defer_to_human'),
            ],
            activations={'r_image_low': low, 'r_image_medium': med, 'r_image_high': high},
            uncertainty=uncertainty,
            trace=trace,
            reduction_loss=_clip01(reduction_loss),
            metadata={
                'adapter': 'medical_image',
                'activation_threshold': self.activation_threshold,
                'modality': modality,
                'protocol': protocol,
                **meta,
            },
        )

