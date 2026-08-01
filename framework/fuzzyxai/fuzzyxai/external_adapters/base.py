from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ExternalPipelineArtifacts:
    """Canonical, read-only view of artifacts produced outside FuzzyXAI."""

    pipeline_id: str
    repository_url: str
    repository_commit: str
    task_type: str
    root: Path
    dataset: dict[str, Any]
    split: dict[str, Any]
    preprocessor: dict[str, Any]
    model: dict[str, Any]
    prediction: dict[str, Any]
    explanation: dict[str, Any]
    run: dict[str, Any]
    evidence_refs: tuple[str, ...]


class ExternalPipelineAdapter(Protocol):
    pipeline_id: str
    repository_commit: str
    task_type: str

    def collect_dataset_artifact(self) -> dict[str, Any]: ...
    def collect_split_artifact(self) -> dict[str, Any]: ...
    def collect_preprocessor_artifact(self) -> dict[str, Any]: ...
    def collect_model_artifact(self) -> dict[str, Any]: ...
    def collect_prediction_artifact(self) -> dict[str, Any]: ...
    def collect_explanation_artifact(self) -> dict[str, Any]: ...
    def build_route_observations(self) -> ExternalPipelineArtifacts: ...
