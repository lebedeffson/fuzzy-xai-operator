"""Typed contracts for the sealed final confirmatory cycle."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InvalidActionDecomposition:
    prediction_failure: bool = False
    route_failure: bool = False
    data_failure: bool = False
    explanation_failure: bool = False
    contract_failure: bool = False

    @property
    def operationally_invalid_automatic_action(self) -> bool:
        return any(self.__dict__.values())

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(name for name, value in self.__dict__.items() if value)


@dataclass(frozen=True)
class ConfirmatoryFeatureVector:
    object_id_hash: str
    predictive: tuple[float | None, ...]
    route: tuple[float | None, ...]
    source_is_oof: bool
    split_id: str

    def __post_init__(self) -> None:
        if len(self.predictive) != 9 or len(self.route) != 13:
            raise ValueError("confirmatory feature vector requires 9 predictive and 13 route channels")
        if not self.source_is_oof:
            raise ValueError("confirmatory controller features must be out-of-fold")
        if any(not 0.0 <= value <= 1.0 for value in (*self.predictive, *self.route) if value is not None):
            raise ValueError("confirmatory feature channels must be normalized to [0, 1]")

    @property
    def missing_channel_count(self) -> int:
        return sum(value is None for value in (*self.predictive, *self.route))


@dataclass(frozen=True)
class ExperimentResultRow:
    experiment_id: str
    hypothesis_id: str
    dataset_id: str
    modality: str
    model_id: str
    explainer_id: str
    policy_id: str
    split_id: str
    seed: int
    object_id_hash: str
    metric: str
    value: float
    source_commit: str
    artifact_sha256: str
