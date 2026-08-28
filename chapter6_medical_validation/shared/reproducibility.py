"""Small, dependency-light runtime facts used by the Chapter 6 bundle."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from .hashing import sha256_file


def environment_facts() -> dict[str, Any]:
    import torch

    try:
        import torchvision
        torchvision_version: str | None = torchvision.__version__
    except Exception as exc:  # a report must disclose, never guess
        torchvision_version = None
        torchvision_error = f"{type(exc).__name__}: {exc}"
    else:
        torchvision_error = None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision_version,
        "torchvision_import_error": torchvision_error, "cuda_available": bool(torch.cuda.is_available()),
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def hash_if_exists(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None
