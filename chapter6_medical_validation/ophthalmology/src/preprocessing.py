from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from .artifact_io import sha256_file, sha256_json


@dataclass(frozen=True)
class PreprocessedImage:
    rgb: np.ndarray
    normalized_chw: np.ndarray
    trace: dict[str, Any]


def crop_black_border(rgb: np.ndarray, threshold: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > int(threshold)
    rows, cols = np.where(mask)
    if not len(rows):
        return rgb.copy(), (0, rgb.shape[0], 0, rgb.shape[1])
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    return rgb[top:bottom, left:right].copy(), (top, bottom, left, right)


def apply_clahe_rgb(rgb: np.ndarray, *, clip_limit: float, grid_size: tuple[int, int]) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(int(v) for v in grid_size))
    enhanced = cv2.merge((clahe.apply(lightness), a, b))
    return cast(np.ndarray, cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB))


def preprocess_image(path: str | Path, config: dict[str, Any], *, split: str, seed: int | None = None) -> PreprocessedImage:
    source = Path(path)
    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("fundus input must decode to HxWx3 RGB")
    operations: list[dict[str, Any]] = [{"op": "load_rgb", "shape": list(rgb.shape)}]
    if bool(config.get("crop_black_border", False)):
        rgb, box = crop_black_border(rgb, int(config.get("black_threshold", 8)))
        operations.append({"op": "crop_black_border", "threshold": int(config.get("black_threshold", 8)), "box": list(box)})
    height, width = (int(v) for v in config["image_size"])
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    operations.append({"op": "resize", "height": height, "width": width, "interpolation": "INTER_AREA"})
    clahe = dict(config.get("clahe", {}))
    if bool(clahe.get("enabled", False)):
        rgb = apply_clahe_rgb(rgb, clip_limit=float(clahe["clip_limit"]), grid_size=tuple(clahe["grid_size"]))
        operations.append({"op": "clahe", **clahe})
    if split == "train":
        if seed is None:
            raise ValueError("train preprocessing requires an explicit deterministic seed")
        rng = np.random.default_rng(seed)
        aug = dict(config.get("train_augmentation", {}))
        if rng.random() < float(aug.get("horizontal_flip_probability", 0.0)):
            rgb = np.ascontiguousarray(rgb[:, ::-1])
            operations.append({"op": "horizontal_flip"})
        angle = float(rng.uniform(-float(aug.get("rotation_degrees", 0.0)), float(aug.get("rotation_degrees", 0.0))))
        if angle:
            matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
            rgb = cv2.warpAffine(rgb, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            operations.append({"op": "rotation", "degrees": angle})
    elif split not in {"validation", "internal_test", "official_test", "test"}:
        raise ValueError(f"unsupported split: {split}")
    values = rgb.astype(np.float32) / 255.0
    normalization = dict(config["normalization"])
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    normalized = ((values - mean) / std).transpose(2, 0, 1)
    if not np.isfinite(normalized).all():
        raise ValueError("preprocessing produced NaN/Inf")
    trace = {
        "source_sha256": sha256_file(source),
        "config_sha256": sha256_json(config),
        "split": split,
        "stochastic_augmentation": split == "train",
        "seed": seed if split == "train" else None,
        "operations": operations,
        "output_shape": list(normalized.shape),
        "output_sha256": sha256_json(normalized.tolist()),
    }
    return PreprocessedImage(rgb=rgb, normalized_chw=normalized.astype(np.float32), trace=trace)
