from __future__ import annotations

import os
from pathlib import Path


def repository_root() -> Path:
    """Locate the source repository without assuming the package is at its root."""

    configured = os.environ.get("FUZZYXAI_REPOSITORY_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if root.exists():
            return root
        raise FileNotFoundError(f"FUZZYXAI_REPOSITORY_ROOT does not exist: {root}")
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "framework/fuzzyxai/fuzzyxai").is_dir() and (candidate / "data").is_dir():
            return candidate
    # A wheel may be imported without the source checkout. Explicit fixture
    # paths remain usable; the default fixture read will fail with its full path.
    return Path(__file__).resolve().parents[1]


def repository_path(relative_path: str | Path) -> Path:
    return repository_root() / Path(relative_path)
