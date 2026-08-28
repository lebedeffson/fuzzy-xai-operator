from __future__ import annotations

import numpy as np
from PIL import Image

from chapter6_medical_validation.ophthalmology.src.preprocessing import preprocess_image

CONFIG = {"image_size": [16, 16], "crop_black_border": True, "black_threshold": 8, "clahe": {"enabled": True, "clip_limit": 2.0, "grid_size": [4, 4]}, "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}, "train_augmentation": {"horizontal_flip_probability": 0.5, "rotation_degrees": 5}}


def test_validation_preprocessing_is_deterministic_and_traceable(tmp_path):
    path = tmp_path / "eye.png"
    image = np.zeros((20, 24, 3), dtype=np.uint8)
    image[2:-2, 3:-3] = [80, 120, 160]
    Image.fromarray(image).save(path)
    first = preprocess_image(path, CONFIG, split="validation")
    second = preprocess_image(path, CONFIG, split="validation")
    assert first.normalized_chw.shape == (3, 16, 16)
    assert np.array_equal(first.normalized_chw, second.normalized_chw)
    assert np.isfinite(first.normalized_chw).all()
    assert first.trace["source_sha256"] == second.trace["source_sha256"]
    assert first.trace["stochastic_augmentation"] is False


def test_train_augmentation_requires_seed(tmp_path):
    path = tmp_path / "eye.png"
    Image.fromarray(np.full((10, 10, 3), 100, dtype=np.uint8)).save(path)
    try:
        preprocess_image(path, CONFIG, split="train")
    except ValueError as exc:
        assert "explicit deterministic seed" in str(exc)
    else:
        raise AssertionError("unseeded training augmentation was accepted")
