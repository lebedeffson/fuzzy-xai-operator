from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .base import ExternalPipelineArtifacts


class ManifestExternalPipelineAdapter:
    """Normalize immutable manifests without interpreting their meaning."""

    MANIFESTS = (
        "dataset",
        "split",
        "preprocessor",
        "model",
        "prediction",
        "explanation",
        "run",
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        registration = self._read("pipeline")
        self.pipeline_id = str(registration["pipeline_id"])
        self.repository_commit = str(registration["repository_commit"])
        self.task_type = str(registration["task_type"])
        self.repository_url = str(registration["repository_url"])

    def _read(self, name: str) -> dict[str, Any]:
        path = self.root / f"{name}_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        return payload

    def collect_dataset_artifact(self) -> dict[str, Any]:
        return self._read("dataset")

    def collect_split_artifact(self) -> dict[str, Any]:
        return self._read("split")

    def collect_preprocessor_artifact(self) -> dict[str, Any]:
        return self._read("preprocessor")

    def collect_model_artifact(self) -> dict[str, Any]:
        return self._read("model")

    def collect_prediction_artifact(self) -> dict[str, Any]:
        return self._read("prediction")

    def collect_explanation_artifact(self) -> dict[str, Any]:
        return self._read("explanation")

    def build_route_observations(self) -> ExternalPipelineArtifacts:
        values = {name: self._read(name) for name in self.MANIFESTS}
        refs = tuple(f"{name}_manifest.json" for name in self.MANIFESTS)
        for ref in refs:
            path = self.root / ref
            if not path.is_file() or not hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"unreadable evidence artifact: {path}")
        return ExternalPipelineArtifacts(
            pipeline_id=self.pipeline_id,
            repository_url=self.repository_url,
            repository_commit=self.repository_commit,
            task_type=self.task_type,
            root=self.root,
            dataset=values["dataset"],
            split=values["split"],
            preprocessor=values["preprocessor"],
            model=values["model"],
            prediction=values["prediction"],
            explanation=values["explanation"],
            run=values["run"],
            evidence_refs=refs,
        )
