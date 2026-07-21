"""Strictly paired local-explainer contracts for H1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LocalExplanation:
    method: str
    object_id: str
    attribution: tuple[float, ...]
    feature_names: tuple[str, ...]
    model_output: float
    background_hash: str
    budget: int
    seed: int
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        if len(self.attribution) != len(self.feature_names):
            raise ValueError("attribution and feature names must align")
        if self.budget <= 0:
            raise ValueError("explainer budget must be positive")


@dataclass(frozen=True)
class WrappedLocalExplanation:
    local: LocalExplanation
    evidence_refs: tuple[str, ...]
    missing_channels: tuple[str, ...]
    traceability: float

    @property
    def attribution(self) -> tuple[float, ...]:
        # The system layer adds evidence and diagnostics. It must not silently
        # alter the local explainer being evaluated in H1.
        return self.local.attribution


def wrap_local_explanation(
    explanation: LocalExplanation,
    *,
    evidence_refs: Sequence[str],
    missing_channels: Sequence[str] = (),
) -> WrappedLocalExplanation:
    refs = tuple(str(item) for item in evidence_refs)
    complete_components = (
        bool(refs),
        bool(explanation.provenance.get("model_sha256")),
        bool(explanation.provenance.get("data_sha256")),
        bool(explanation.provenance.get("method_version")),
    )
    return WrappedLocalExplanation(
        local=explanation,
        evidence_refs=refs,
        missing_channels=tuple(sorted(set(missing_channels))),
        traceability=sum(complete_components) / len(complete_components),
    )


def deletion_fidelity(
    *,
    predict_probability: object,
    sample: np.ndarray,
    reference: np.ndarray,
    attribution: Sequence[float],
    top_k: int,
) -> float:
    if sample.ndim != 1 or reference.shape != sample.shape or len(attribution) != len(sample):
        raise ValueError("deletion fidelity inputs must align")
    if not callable(predict_probability):
        raise TypeError("predict_probability must be callable")
    top = np.argsort(np.abs(np.asarray(attribution, dtype=float)))[-max(1, min(top_k, len(sample))) :]
    deleted = sample.copy()
    deleted[top] = reference[top]
    before = float(np.asarray(predict_probability(sample.reshape(1, -1))).reshape(-1)[0])
    after = float(np.asarray(predict_probability(deleted.reshape(1, -1))).reshape(-1)[0])
    return abs(before - after)
