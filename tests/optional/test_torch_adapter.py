from __future__ import annotations

import numpy as np
import pytest
from fuzzyxai import FuzzyXAI
from fuzzyxai.adapters.contracts_v2 import ExplanationContext
from fuzzyxai.adapters.optional_v2 import TorchAdapter


@pytest.mark.optional_integration
def test_torch_integrated_gradients_and_mode_restoration() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(3, 2))
    model.train()
    values = np.asarray([[0.2, -0.1, 0.7]], dtype=np.float32)
    result = FuzzyXAI.wrap(model).explain_one(values, feature_names=("a", "b", "c"))
    assert result.adapter_id == "torch_v2"
    assert result.model_evidence["contribution_method"] == "derived_native_integrated_gradients"
    assert np.isfinite(list(result.model_evidence["contributions"].values())).all()
    assert model.training is True


@pytest.mark.optional_integration
def test_torch_ig_keeps_one_target_when_argmax_changes_along_path() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 2)
    with torch.no_grad():
        model.weight[:] = torch.tensor([[-2.0], [2.0]])
        model.bias[:] = torch.tensor([1.0, 0.0])
    adapter = TorchAdapter(model, task="classification", ig_steps=32)
    values = np.asarray([[1.0]], dtype=np.float32)
    prediction = adapter.predict(values)
    evidence = adapter.extract_local_evidence(values, prediction, ExplanationContext(feature_names=("x",)))
    completeness = evidence.channels["ig_completeness"]
    assert completeness["target_class"] == 1
    assert completeness["n_steps"] == 32
    assert completeness["integration_points"] == 33
    assert completeness["input_output_delta"] == pytest.approx(2.0)
    assert completeness["attribution_sum"] == pytest.approx(2.0, abs=1e-6)
    assert completeness["completeness_residual"] < 1e-6


@pytest.mark.optional_integration
def test_ig_completeness_is_projected_into_public_reader_report() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    values = np.asarray([[0.2, -0.1, 0.7, 0.4]], dtype=np.float32)
    image = values.reshape(2, 2)
    adapter = TorchAdapter(
        model,
        task="classification",
        input_transform=lambda raw: torch.as_tensor(raw, dtype=torch.float32).reshape(-1, 1, 2, 2),
    )
    result = FuzzyXAI.wrap(model, adapter=adapter).explain_one(
        values,
        raw_object=image,
        feature_names=("px0", "px1", "px2", "px3"),
    )
    report = result.full_report(level="reader")
    completeness = result.view_model.layers["attribution_maps"][0]["completeness"]
    assert completeness["status"] == "measured"
    assert "IG completeness" in report
    assert "absolute residual=" in report
    assert "relative residual=" in report
