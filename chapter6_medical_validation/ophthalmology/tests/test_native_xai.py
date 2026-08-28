from __future__ import annotations

import numpy as np
import torch

from chapter6_medical_validation.ophthalmology.src.native_xai import grad_cam


class TinyConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Conv2d(3, 2, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.head = torch.nn.Linear(2, 5)

    def forward(self, value):
        return self.head(self.pool(torch.relu(self.features(value))).flatten(1))


def test_grad_cam_tracks_target_sample_checkpoint_and_shape():
    torch.manual_seed(3)
    model = TinyConv()
    result = grad_cam(model, torch.ones((1, 3, 8, 8)), model.features, target_layer_id="features", sample_id="eye-1", checkpoint_sha256="a" * 64, target_class=2)
    assert result.target_class == 2
    assert result.sample_id == "eye-1"
    assert result.checkpoint_sha256 == "a" * 64
    assert result.raw_map.shape == (8, 8)
    assert np.isfinite(result.raw_map).all() and result.raw_map.size > 0
