from __future__ import annotations

import numpy as np
import torch
from fuzzyxai.adapters.contracts_v2 import ExplanationContext

from chapter6_medical_validation.shared.torch_adapter import TemperatureScaledTorchAdapter


def test_medical_tensor_ig_does_not_duplicate_flat_contributions() -> None:
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    adapter = TemperatureScaledTorchAdapter(model, temperature=1.0, task="classification", ig_steps=4)
    inputs = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    prediction = adapter.predict(inputs)

    evidence = adapter.extract_local_evidence(
        inputs,
        prediction,
        ExplanationContext(target=prediction.predictions[0]),
    )

    assert "integrated_gradients" in evidence.channels
    assert "ig_completeness" in evidence.channels
    assert "contributions" not in evidence.channels
    assert evidence.channels["flat_contributions_status"].startswith("not_materialized")
