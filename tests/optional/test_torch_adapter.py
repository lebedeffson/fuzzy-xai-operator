from __future__ import annotations

import numpy as np
import pytest

from fuzzyxai import FuzzyXAI


@pytest.mark.optional_integration
def test_torch_integrated_gradients_and_mode_restoration() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(3, 2))
    model.train()
    values = np.asarray([[0.2, -0.1, 0.7]], dtype=np.float32)
    result = FuzzyXAI.wrap(model).explain_one(values, feature_names=("a", "b", "c"))
    assert result.adapter.adapter_id == "torch_v2"
    assert result.model_evidence["contribution_method"] == "derived_native_integrated_gradients"
    assert result.model_evidence["gradient_sanity"] is True
    assert model.training is True
