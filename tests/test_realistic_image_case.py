"""P12: one realistic image case, end-to-end.

Uses a real trained CNN (not a synthetic/toy formula) on sklearn's bundled
digits dataset, real Integrated Gradients attribution aggregated into named
regions, and asserts the explanation reads as a handful of meaningful
regions rather than being buried under one claim per pixel — the exact
gap identified when the image pipeline first shipped.
"""

from __future__ import annotations

import numpy as np
import pytest
from fuzzyxai import FuzzyXAI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

try:
    import torch

    HAS_TORCH = True
    del torch
except ImportError:
    HAS_TORCH = False

REGIONS = ("top_left", "top_right", "bottom_left", "bottom_right")


def _region_mask(name: str) -> np.ndarray:
    mask = np.zeros((8, 8), dtype=bool)
    row0 = 0 if "top" in name else 4
    col0 = 0 if "left" in name else 4
    mask[row0 : row0 + 4, col0 : col0 + 4] = True
    return mask


def _region_of_pixel(index: int) -> str:
    row, col = divmod(index, 8)
    return f"{'top' if row < 4 else 'bottom'}_{'left' if col < 4 else 'right'}"


def _build_case():
    import torch
    from fuzzyxai.adapters.optional_v2 import TorchAdapter
    from torch import nn

    torch.manual_seed(0)
    digits = load_digits()
    is_3_or_8 = np.isin(digits.target, [3, 8])
    images = digits.images[is_3_or_8].astype(np.float32) / 16.0
    labels = (digits.target[is_3_or_8] == 8).astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=0, stratify=labels)

    class DigitCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())
            self.fc = nn.Linear(16 * 8 * 8, 2)

        def forward(self, x):
            return self.fc(self.conv(x).flatten(1))

    net = DigitCNN()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    x_train_tensor = torch.as_tensor(X_train).unsqueeze(1)
    y_train_tensor = torch.as_tensor(y_train)
    for _ in range(40):
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
    pixel_result = fx.explain_one(flat, feature_names=[f"px_{i}" for i in range(64)])
    pixel_contributions = dict(pixel_result.model_evidence["contributions"])
    region_sums = {region: 0.0 for region in REGIONS}
    for index in range(64):
        region_sums[_region_of_pixel(index)] += pixel_contributions.get(f"px_{index}", 0.0)

    result = fx.explain_one(
        flat,
        raw_object=sample_image,
        region_masks={name: _region_mask(name) for name in REGIONS},
        evidence={"contributions": region_sums},
    )
    return net, X_test, y_test, result


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_native_prediction_matches() -> None:
    import torch

    net, X_test, _, result = _build_case()
    with torch.no_grad():
        native = int(torch.argmax(net(torch.as_tensor(X_test[:1]).unsqueeze(1)), dim=-1).item())
    assert result.prediction.predictions[0] == native


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_claim_count_stays_small_not_one_per_pixel() -> None:
    _, _, _, result = _build_case()
    # 64 pixels would mean 64+ feature_contribution claims if not aggregated
    # into regions; the realistic case must stay well below that.
    assert len(result.claims) < 20


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_no_duplicate_evidence_between_feature_contribution_and_image_region() -> None:
    """A region's aggregated value must not appear as both a generic
    feature_contribution claim and an image_region claim under the same name."""

    _, _, _, result = _build_case()
    feature_contribution_subjects = {c.subject_id for c in result.claims if c.claim_type == "feature_contribution"}
    image_region_subjects = {c.subject_id for c in result.claims if c.claim_type == "image_region"}
    assert not (feature_contribution_subjects & image_region_subjects)
    assert image_region_subjects == set(REGIONS)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_all_four_regions_have_measured_not_unknown_contributions() -> None:
    _, _, _, result = _build_case()
    region_claims = [c for c in result.claims if c.claim_type == "image_region"]
    assert len(region_claims) == 4
    for claim in region_claims:
        assert claim.metric_value is not None
        assert claim.effect in {"favorable", "adverse"}


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_summary_names_regions_not_raw_pixel_indices() -> None:
    _, _, _, result = _build_case()
    text = result.summary()
    assert any(region in text for region in REGIONS)
    assert "px_" not in text


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_object_representation_is_image_modality_with_real_regions() -> None:
    _, _, _, result = _build_case()
    repr_ = result.object_representation
    assert repr_["modality"] == "image"
    assert repr_["image_width"] == 8
    assert repr_["image_height"] == 8
    assert len(repr_["image_regions"]) == 4


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_visualization_renders_headless() -> None:
    _, _, _, result = _build_case()
    figure = result.visualize(view="object_representation", backend="matplotlib")
    assert figure is not None
