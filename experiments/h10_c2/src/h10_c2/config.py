from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIG_DIR


def load_yaml(name_or_path: str | Path) -> dict[str, Any]:
    path = Path(name_or_path)
    if not path.exists():
        path = CONFIG_DIR / str(name_or_path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value

