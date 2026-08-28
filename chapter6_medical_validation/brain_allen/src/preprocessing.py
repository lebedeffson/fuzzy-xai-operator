from __future__ import annotations

import numpy as np


def preprocess_patch(patch: np.ndarray, *, scale: float, split: str, seed: int, object_index: int):
    import torch

    image = torch.tensor(np.asarray(patch) / scale, dtype=torch.float32).clamp(0, 1)[None, None]
    image = torch.nn.functional.interpolate(image, size=(299, 299), mode="bilinear", align_corners=False)[0].repeat(3, 1, 1)
    operations = ["train_intensity_scale", "clip_0_1", "resize_299_bilinear", "repeat_grayscale_3ch"]
    if split == "train":
        generator = torch.Generator().manual_seed(seed * 1_000_003 + object_index)
        if torch.rand((), generator=generator) < 0.5:
            image = torch.flip(image, dims=(2,)); operations.append("horizontal_flip")
    elif split not in {"validation", "test"}:
        raise ValueError(f"unknown brain split: {split}")
    display = (image[0].numpy() * 255).astype(np.uint8)
    # The frozen three-seed runs used a symmetric [-1, 1] transform. Keep the
    # replay contract exact: substituting ImageNet RGB statistics after
    # training changes the learned decision.
    image = (image - 0.5) / 0.5
    operations.append("symmetric_normalization_mean_0.5_std_0.5")
    return image, display, {"version": "allen_nissl_preprocess_v1", "split": split, "scale_train_p99_5": scale, "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}, "operations": operations, "stochastic": split == "train"}
