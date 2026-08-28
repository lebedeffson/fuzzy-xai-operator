from __future__ import annotations

import numpy as np
import torch

from chapter6_medical_validation.ecg_ptbxl.src.model import build_ecg_resnet1d
from chapter6_medical_validation.ecg_ptbxl.src.preprocessing import fit_lead_statistics, normalize
from chapter6_medical_validation.ecg_ptbxl.src.xai import common_ig_representation, common_occlusion_representation, temporal_occlusion


def test_ecg_model_probability_and_registered_shapes():
    torch.manual_seed(2026)
    model = build_ecg_resnet1d(channels=(4, 8), blocks_per_stage=1).eval()
    value = torch.zeros(2, 12, 1000)
    logits = model(value)
    probabilities = torch.softmax(logits, dim=1)
    assert logits.shape == (2, 2)
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(probabilities.sum(1), torch.ones(2))


def test_normalization_is_train_fitted_and_xai_grids_match():
    signals = [np.ones((12, 1000)), np.full((12, 1000), 3.0)]
    stats = fit_lead_statistics(signals)
    assert stats["status"] == "fitted_on_train_only"
    assert np.allclose(normalize(signals[0], stats), -1.0)
    ig = common_ig_representation(np.arange(12000).reshape(12, 1000))
    occ = common_occlusion_representation(np.ones((12, 20)))
    assert ig.shape == occ.shape == (12, 20)
    assert np.isclose(np.abs(ig).sum(), 1.0)
    assert np.isclose(np.abs(occ).sum(), 1.0)


def test_temporal_occlusion_has_signed_12_by_20_result():
    model = build_ecg_resnet1d(channels=(4,), blocks_per_stage=1).eval()
    result = temporal_occlusion(model, torch.ones(1, 12, 1000), target=1, temperature=2.0, window=50, stride=50)
    assert result["importance"].shape == (12, 20)
    assert result["baseline"] == "zero_standardized_train_mean"
    assert result["temperature"] == 2.0
