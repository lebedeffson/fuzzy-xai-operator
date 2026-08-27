"""P15.2: AttributionMapEvidence — the full per-pixel attribution tensor is
preserved and rendered as a real overlay, never collapsed into a handful of
arbitrary quadrants. `TorchAdapter` already computes the full Integrated
Gradients tensor; before this it was silently dropped by runtime.py, which
only ever read the flattened per-pixel `contributions` dict.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch

    HAS_TORCH = True
    del torch
except ImportError:
    HAS_TORCH = False


def _build_case():
    import torch
    from fuzzyxai import FuzzyXAI
    from fuzzyxai.adapters.optional_v2 import TorchAdapter
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from torch import nn

    torch.manual_seed(0)
    digits = load_digits()
    is_3_or_8 = np.isin(digits.target, [3, 8])
    images = digits.images[is_3_or_8].astype(np.float32) / 16.0
    labels = (digits.target[is_3_or_8] == 8).astype(np.int64)
    X_train, X_test, y_train, _ = train_test_split(images, labels, test_size=0.2, random_state=0, stratify=labels)

    class DigitCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU())
            self.fc = nn.Linear(4 * 8 * 8, 2)

        def forward(self, x):
            return self.fc(self.conv(x).flatten(1))

    net = DigitCNN()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
    loss_fn = nn.CrossEntropyLoss()
    x_train_tensor = torch.as_tensor(X_train).unsqueeze(1)
    y_train_tensor = torch.as_tensor(y_train)
    for _ in range(15):
        optimizer.zero_grad()
        loss_fn(net(x_train_tensor), y_train_tensor).backward()
        optimizer.step()
    net.eval()

    def input_transform(x):
        return torch.as_tensor(np.asarray(x, dtype=np.float32).reshape(-1, 1, 8, 8))

    adapter = TorchAdapter(net, task="classification", input_transform=input_transform)
    fx = FuzzyXAI.wrap(net, adapter=adapter)

    sample_image = X_test[0]
    flat = sample_image.flatten().tolist()
    result = fx.explain_one(flat, raw_object=sample_image, feature_names=[f"px_{i}" for i in range(64)])
    return result


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_attribution_map_preserves_full_tensor_not_a_handful_of_quadrants() -> None:
    result = _build_case()
    maps = result.view_model.layers["attribution_maps"]
    assert len(maps) == 1
    attribution = maps[0]
    assert attribution["method"] == "integrated_gradients"
    assert tuple(attribution["shape"]) not in {(4,), (2, 2)}  # not collapsed to a handful of quadrant numbers
    array = np.asarray(attribution["attribution_array"])
    assert array.size >= 64  # full per-pixel resolution preserved, not aggregated away
    assert attribution["attribution_png_base64"]


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_attribution_map_summary_stats_are_measured_not_guessed() -> None:
    result = _build_case()
    attribution = result.view_model.layers["attribution_maps"][0]
    array = np.asarray(attribution["attribution_array"]).sum(axis=0)  # channel-first sum, matches build logic
    assert attribution["min_value"] == pytest.approx(float(array.min()), abs=1e-4)
    assert attribution["max_value"] == pytest.approx(float(array.max()), abs=1e-4)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_attribution_map_is_reachable_and_claimed() -> None:
    result = _build_case()
    assert result.explanation_graph.validate_reachability() == ()
    claim_types = {claim["claim_type"] for claim in result.view_model.claims}
    assert "attribution_map" in claim_types


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_object_representation_carries_attribution_overlay() -> None:
    result = _build_case()
    repr_ = result.object_representation
    assert repr_["modality"] == "image"
    assert repr_["attribution_overlay_png_base64"]
    assert repr_["attribution_method"] == "integrated_gradients"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_raw_attribution_bytes_redacted_by_default_but_available_on_request() -> None:
    result = _build_case()
    redacted = result.to_dict(include_raw=False)
    full = result.to_dict(include_raw=True)
    redacted_array = redacted["layers"]["attribution_maps"][0]["attribution_array"]
    full_array = full["layers"]["attribution_maps"][0]["attribution_array"]
    assert redacted_array != full_array
    assert isinstance(full_array, list)


def test_tabular_case_without_raw_object_has_no_attribution_maps_fabricated() -> None:
    from fuzzyxai import FuzzyXAI
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression

    X, y = load_breast_cancer(return_X_y=True)
    model = LogisticRegression(max_iter=2000).fit(X, y)
    result = FuzzyXAI.wrap(model).explain_one(X[0], object_id="p0")
    assert result.view_model.layers["attribution_maps"] == []
