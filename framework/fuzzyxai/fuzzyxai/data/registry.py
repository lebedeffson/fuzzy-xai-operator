from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from fuzzyxai.audit.common import ROOT


DATASETS_YAML = ROOT / "data" / "registry" / "datasets.yaml"


def load_dataset_registry(path: Path = DATASETS_YAML) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Dataset registry must be a mapping")
    return data


def dataset_entry(scenario_id: str) -> dict[str, Any]:
    registry = load_dataset_registry()
    if scenario_id not in registry:
        raise KeyError(scenario_id)
    return registry[scenario_id]

