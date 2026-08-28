from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GradCAMResult:
    raw_map: np.ndarray
    normalized_map: np.ndarray
    target_class: int
    target_layer: str
    sample_id: str
    checkpoint_sha256: str
    output_space: str = "logit"

    def metadata(self) -> dict[str, Any]:
        return {
            "method": "grad_cam",
            "target_class": self.target_class,
            "target_layer": self.target_layer,
            "sample_id": self.sample_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "output_space": self.output_space,
            "shape": list(self.raw_map.shape),
            "finite": bool(np.isfinite(self.raw_map).all()),
        }


def grad_cam(
    model: Any,
    input_tensor: Any,
    target_layer: Any,
    *,
    target_layer_id: str,
    sample_id: str,
    checkpoint_sha256: str,
    target_class: int | None = None,
) -> GradCAMResult:
    """Standard experiment-side Grad-CAM; no FuzzyXAI core computation."""

    import torch
    from torch.nn import functional

    activations: list[Any] = []
    gradients: list[Any] = []

    def forward_hook(_module: Any, _inputs: Any, output: Any) -> None:
        activations.append(output)

    def backward_hook(_module: Any, _grad_input: Any, grad_output: Any) -> None:
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    was_training = bool(model.training)
    model.eval()
    try:
        model.zero_grad(set_to_none=True)
        working_input = input_tensor.detach().clone().requires_grad_(True)
        logits = model(working_input)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("Grad-CAM currently requires one N=1 classification tensor")
        selected = int(torch.argmax(logits[0]).item()) if target_class is None else int(target_class)
        if not 0 <= selected < logits.shape[1]:
            raise ValueError("Grad-CAM target class is outside model output")
        logits[0, selected].backward()
        if not activations or not gradients:
            raise RuntimeError("target layer hooks produced no activations/gradients")
        feature_map, gradient = activations[-1], gradients[-1]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))
        cam = functional.interpolate(cam, size=working_input.shape[-2:], mode="bilinear", align_corners=False)
        raw = cam[0, 0].detach().cpu().numpy().astype(np.float64)
        maximum = float(raw.max())
        normalized = raw / maximum if maximum > 0 else np.zeros_like(raw)
        if not np.isfinite(raw).all() or raw.size == 0:
            raise ValueError("Grad-CAM map is empty or non-finite")
        return GradCAMResult(raw, normalized, selected, target_layer_id, sample_id, checkpoint_sha256)
    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
