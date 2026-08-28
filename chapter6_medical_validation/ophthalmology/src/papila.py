"""PAPILA-specific factual preprocessing, kept outside the FuzzyXAI core."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def expert1_disc_roi(image_path: Path, contour_path: Path, *, margin_fraction: float = 0.20) -> np.ndarray:
    """Extract a deterministic optic-disc ROI using only expert-1 contour data."""
    points = np.loadtxt(contour_path, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError(f"invalid expert-1 optic-disc contour: {contour_path}")
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    x0, y0 = points.min(axis=0); x1, y1 = points.max(axis=0)
    side = max(float(x1 - x0), float(y1 - y0)) * (1.0 + 2.0 * margin_fraction)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    left, top = int(np.floor(cx - side / 2)), int(np.floor(cy - side / 2))
    right, bottom = int(np.ceil(cx + side / 2)), int(np.ceil(cy + side / 2))
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - rgb.shape[1]), max(0, bottom - rgb.shape[0])
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        rgb = cv2.copyMakeBorder(rgb, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
        left += pad_left; right += pad_left; top += pad_top; bottom += pad_top
    roi = rgb[top:bottom, left:right]
    if roi.size == 0:
        raise ValueError(f"empty expert-1 ROI for {image_path}")
    return np.ascontiguousarray(roi)


def contour_masks_in_registered_roi(image_path: Path, expert1_disc_path: Path, contour_paths: dict[str, Path], *, margin_fraction: float = 0.20, size: int = 224) -> dict[str, np.ndarray]:
    """Rasterize original-coordinate contours in the same expert-1 ROI frame."""
    with Image.open(image_path) as image:
        height, width = image.height, image.width
    base = np.loadtxt(expert1_disc_path, dtype=np.float64); x0, y0 = base.min(axis=0); x1, y1 = base.max(axis=0); side = max(float(x1-x0), float(y1-y0)) * (1 + 2 * margin_fraction); cx, cy = (x0+x1)/2, (y0+y1)/2
    left, top = int(np.floor(cx-side/2)), int(np.floor(cy-side/2)); right, bottom = int(np.ceil(cx+side/2)), int(np.ceil(cy+side/2)); pad_left, pad_top = max(0,-left), max(0,-top)
    canvas_shape=(height+max(0,bottom-height)+pad_top,width+max(0,right-width)+pad_left)
    left += pad_left; right += pad_left; top += pad_top; bottom += pad_top
    masks: dict[str,np.ndarray]={}
    for name,path in contour_paths.items():
        points=np.loadtxt(path,dtype=np.float64); points[:,0]+=pad_left; points[:,1]+=pad_top; mask=np.zeros(canvas_shape,dtype=np.uint8); cv2.fillPoly(mask,[np.round(points).astype(np.int32)],1); masks[name]=cv2.resize(mask[top:bottom,left:right],(size,size),interpolation=cv2.INTER_NEAREST).astype(bool)
    return masks


def papila_tensor(image_path: Path, contour_path: Path, config: dict[str, Any], *, training: bool, seed: int | None) -> np.ndarray:
    """Registered ROI -> 224x224 ImageNet tensor; no diagnosis enters the crop."""
    roi = expert1_disc_roi(image_path, contour_path, margin_fraction=float(config.get("roi_margin_fraction", 0.20)))
    height, width = (int(value) for value in config.get("image_size", [224, 224]))
    rgb = cv2.resize(roi, (width, height), interpolation=cv2.INTER_AREA)
    if training:
        if seed is None:
            raise ValueError("training PAPILA preprocessing requires an explicit seed")
        rng = np.random.default_rng(seed)
        if rng.random() < float(config.get("horizontal_flip_probability", 0.5)):
            rgb = np.ascontiguousarray(rgb[:, ::-1])
        degrees = float(config.get("rotation_degrees", 8.0))
        angle = float(rng.uniform(-degrees, degrees))
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        rgb = cv2.warpAffine(rgb, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    values = rgb.astype(np.float32) / 255.0
    mean = np.asarray(config.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.asarray(config.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
    return ((values - mean) / std).transpose(2, 0, 1).astype(np.float32)
