from __future__ import annotations

from .types import AdaptedInput, ExplainableObject


def build_explainable_object(adapted: AdaptedInput) -> ExplainableObject:
    """Build E_k from an adapted external-model payload."""

    values = adapted.values
    components = {
        "model_signal": values.get("model_match_signal"),
        "image_quality": values.get("image_quality"),
        "segmentation_quality": values.get("segmentation_quality"),
        "alpha_accept": values.get("alpha_accept"),
        "alpha_block": values.get("alpha_block"),
    }
    return ExplainableObject(
        scenario_id=adapted.scenario_id,
        adapted_input=adapted,
        components={key: value for key, value in components.items() if value is not None},
        metadata={"source": adapted.source, **adapted.metadata},
    )
