from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json_once(path: str | Path, payload: Any) -> None:
    target = Path(path)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"frozen artifact differs and will not be overwritten: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(encoded, encoding="utf-8")


def relative_inventory(root: str | Path, paths: list[Path]) -> list[dict[str, Any]]:
    base = Path(root).resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted((item.resolve() for item in paths), key=lambda item: item.as_posix()):
        try:
            relative = path.relative_to(base).as_posix()
        except ValueError as exc:
            raise ValueError(f"inventory path escapes data root: {path}") from exc
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def environment_manifest() -> dict[str, Any]:
    torch_data: dict[str, Any]
    try:
        import torch

        torch_data = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - binary/runtime disclosure
        torch_data = {"status": "not_available", "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "python": sys.version,
        "os": platform.platform(),
        "machine": platform.machine(),
        "torch_runtime": torch_data,
        "data_root_source": "FUZZYXAI_CH6_DATA_ROOT",
        "data_root_configured": bool(os.environ.get("FUZZYXAI_CH6_DATA_ROOT")),
    }
