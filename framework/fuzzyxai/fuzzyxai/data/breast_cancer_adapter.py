from __future__ import annotations

from datetime import datetime, timezone

from fuzzyxai.core.explanation_object import ExplanationObject, Rule, Trace
from fuzzyxai.hierarchy.f0 import F0


def build_explanation_for_prediction(
    prob_malignant: float,
    *,
    sample_id: str,
    model_version: str = 'rf_v1',
    dataset_id: str = 'sklearn_breast_cancer',
    reduction_loss: float = 0.0,
) -> ExplanationObject:
    """Build a concrete explanation object for one breast-cancer prediction."""
    p = max(0.0, min(1.0, float(prob_malignant)))
    low_activation = 1.0 - p
    high_activation = p
    uncertainty = 1.0 - abs(high_activation - low_activation)

    rules = [
        Rule('r_low_risk', {'prob_malignant': 'low'}, 'low_risk'),
        Rule('r_high_risk', {'prob_malignant': 'high'}, 'high_risk'),
    ]
    activations = {
        'r_low_risk': float(low_activation),
        'r_high_risk': float(high_activation),
    }
    trace = Trace(
        id=sample_id,
        version=model_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        source=dataset_id,
        checksum=f'{sample_id}:{model_version}',
    )
    return ExplanationObject(
        terms={'low_risk', 'high_risk'},
        representation=F0(lambda _x, level=p: level, label='malignant_risk'),
        rules=rules,
        activations=activations,
        uncertainty=float(uncertainty),
        trace=trace,
        reduction_loss=max(0.0, min(1.0, float(reduction_loss))),
        metadata={'activation_threshold': 0.05, 'prob_malignant': p},
    )
