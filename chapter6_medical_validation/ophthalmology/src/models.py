from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def _torchvision_models() -> Any:
    try:
        from torchvision import models
    except Exception as exc:  # pragma: no cover - depends on local binary compatibility
        raise RuntimeError(
            "torchvision cannot be imported in this environment. Install a torchvision "
            "build compatible with the installed torch before medical training; current "
            f"error: {type(exc).__name__}: {exc}"
        ) from exc
    return models


def build_classifier(architecture: str, *, num_classes: int = 5, pretrained: bool = True) -> Any:
    from torch import nn

    models = _torchvision_models()
    if architecture == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"unsupported registered architecture: {architecture}")


def resolve_module(model: Any, dotted_path: str) -> Any:
    current = model
    for part in dotted_path.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def model_fingerprint(model: Any) -> str:

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(model: Any, path: str | Path, metadata: dict[str, Any]) -> str:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return digest
