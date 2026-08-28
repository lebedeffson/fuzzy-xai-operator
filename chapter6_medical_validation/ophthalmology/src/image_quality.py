from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def technical_image_quality(rgb: np.ndarray, *, dark_threshold: int = 16, bright_threshold: int = 240) -> dict[str, Any]:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("technical image quality expects RGB HxWx3")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    non_black = gray > dark_threshold
    return {
        "status": "measured",
        "semantics": "technical_image_quality_evidence_not_clinical_quality",
        "blur_laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "underexposure_fraction": float(np.mean(gray <= dark_threshold)),
        "overexposure_fraction": float(np.mean(gray >= bright_threshold)),
        "field_of_view_coverage": float(np.mean(non_black)),
        "parameters": {"dark_threshold": dark_threshold, "bright_threshold": bright_threshold},
    }


def quality_score_from_validation_reference(evidence: dict[str, Any], reference: dict[str, float]) -> float:
    """Declared technical score whose thresholds must come from train/validation."""

    blur_floor = float(reference["blur_floor"])
    max_under = float(reference["max_underexposure_fraction"])
    max_over = float(reference["max_overexposure_fraction"])
    min_fov = float(reference["min_field_of_view_coverage"])
    penalties = [
        1.0 if float(evidence["blur_laplacian_variance"]) < blur_floor else 0.0,
        min(1.0, float(evidence["underexposure_fraction"]) / max(max_under, 1e-12)),
        min(1.0, float(evidence["overexposure_fraction"]) / max(max_over, 1e-12)),
        min(1.0, max(0.0, min_fov - float(evidence["field_of_view_coverage"])) / max(min_fov, 1e-12)),
    ]
    return float(1.0 - max(penalties))
