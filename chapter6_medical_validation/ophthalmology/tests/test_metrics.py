from __future__ import annotations

import numpy as np

from chapter6_medical_validation.ophthalmology.src.metrics import attribution_spatial_metrics, classification_metrics, deletion_faithfulness


def test_classification_and_spatial_metrics_are_explicit():
    truth = np.asarray([0, 1, 2, 3, 4])
    probabilities = np.eye(5) * 0.8 + 0.04
    metrics = classification_metrics(truth, probabilities)
    assert metrics["accuracy"] == 1.0
    heat = np.zeros((4, 4)); heat[1, 1] = 1.0
    mask = np.zeros((4, 4), dtype=bool); mask[1, 1] = True
    spatial = attribution_spatial_metrics(heat, mask)
    assert spatial["pointing_game_inside"] is True
    assert "not_causality" in spatial["semantics"]


def test_deletion_uses_registered_random_control():
    image = np.ones((4, 4, 3), dtype=float)
    heat = np.arange(16, dtype=float).reshape(4, 4)
    def predict(batch):
        score = np.asarray(batch).mean(axis=(1, 2, 3))
        result = np.zeros((len(score), 5)); result[:, 0] = score; result[:, 1:] = (1 - score[:, None]) / 4
        return result
    result = deletion_faithfulness(image, heat, predict, target=0, random_repeats=5)
    assert result["random_repeats"] == 5
    assert result["semantics"].startswith("controlled_deletion")
