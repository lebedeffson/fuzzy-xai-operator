from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contracts import ModelInternalsEvidence

_KNOWN_CHANNELS = (
    "coefficients",
    "intercept",
    "linear_terms",
    "reconstructed_score",
    "actual_score",
    "reconstruction_error",
    "pipeline_steps",
    "decision_path",
    "leaf_id",
    "leaf_samples",
    "ensemble_votes",
    "ensemble_disagreement",
    "global_importance",
)


def collect_model_internals(
    internal_evidence: Mapping[str, Any],
    *,
    object_id: str,
    model_family: str,
) -> ModelInternalsEvidence | None:
    """Surface the model-family-specific internal evidence an adapter's
    ``extract_local_evidence`` already computed (linear coefficients/intercept,
    tree decision path/leaf, ensemble votes/disagreement, global importance)
    instead of discarding it after it was used only to derive the generic
    ``contributions`` map (P15.1/P15.3).

    Returns None when the adapter genuinely exposed none of these channels
    for this object — never a populated-but-empty placeholder.
    """

    present = {key: internal_evidence[key] for key in _KNOWN_CHANNELS if internal_evidence.get(key) is not None}
    if not present:
        return None
    limitations = tuple(str(item) for item in internal_evidence.get("limitations", ()))
    return ModelInternalsEvidence(
        object_id=str(object_id),
        model_family=str(model_family),
        coefficients=present.get("coefficients"),
        intercept=present.get("intercept"),
        linear_terms=present.get("linear_terms"),
        reconstructed_score=present.get("reconstructed_score"),
        actual_score=present.get("actual_score"),
        reconstruction_error=present.get("reconstruction_error"),
        pipeline_steps=present.get("pipeline_steps"),
        decision_path=present.get("decision_path"),
        leaf_id=present.get("leaf_id"),
        leaf_samples=present.get("leaf_samples"),
        ensemble_votes=present.get("ensemble_votes"),
        ensemble_disagreement=present.get("ensemble_disagreement"),
        global_importance=present.get("global_importance"),
        limitations=limitations,
    )
