"""A realistic image case: a real trained CNN, explained end-to-end (P12).

    python examples/04_image_explanation.py

Distinguishes handwritten "3" from "8" (sklearn's bundled digit dataset, no
download needed) with a small real convolutional network — genuinely
trained, not a toy formula. Its real Integrated Gradients attribution (the
same mechanism examples/06 and the P4.3 cross-model tests exercise) is
aggregated into four named quadrant regions before being explained, so the
result reads as a handful of meaningful regions rather than being buried
under one contribution claim per pixel — the exact problem flagged when the
image pipeline first shipped with a flat 1800-pixel synthetic example.
"""

from __future__ import annotations

import numpy as np
import torch
from fuzzyxai import FuzzyXAI
from fuzzyxai.adapters.optional_v2 import TorchAdapter
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

REGIONS = ("top_left", "top_right", "bottom_left", "bottom_right")


class DigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Linear(16 * 8 * 8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


def _region_mask(name: str) -> np.ndarray:
    mask = np.zeros((8, 8), dtype=bool)
    row0 = 0 if "top" in name else 4
    col0 = 0 if "left" in name else 4
    mask[row0 : row0 + 4, col0 : col0 + 4] = True
    return mask


def _region_of_pixel(index: int) -> str:
    row, col = divmod(index, 8)
    vertical = "top" if row < 4 else "bottom"
    horizontal = "left" if col < 4 else "right"
    return f"{vertical}_{horizontal}"


def main() -> None:
    torch.manual_seed(0)
    digits = load_digits()
    is_3_or_8 = np.isin(digits.target, [3, 8])
    images = digits.images[is_3_or_8].astype(np.float32) / 16.0
    labels = (digits.target[is_3_or_8] == 8).astype(np.int64)  # 0 = "3", 1 = "8"
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=0, stratify=labels)

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

    with torch.no_grad():
        test_accuracy = (torch.argmax(net(torch.as_tensor(X_test).unsqueeze(1)), dim=-1).numpy() == y_test).mean()
    print(f"real trained CNN, held-out test accuracy: {test_accuracy:.1%}")

    def input_transform(x: object) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x, dtype=np.float32).reshape(-1, 1, 8, 8))

    adapter = TorchAdapter(net, task="classification", input_transform=input_transform)
    fx = FuzzyXAI.wrap(net, adapter=adapter)

    sample_image = X_test[0]
    flat = sample_image.flatten().tolist()

    # Pass 1: get the real per-pixel Integrated Gradients the adapter computes.
    pixel_result = fx.explain_one(flat, feature_names=[f"px_{i}" for i in range(64)])
    pixel_contributions = dict(pixel_result.model_evidence["contributions"])

    # Aggregate into 4 named quadrant regions — a real sum of real per-pixel
    # attributions, not a fabricated region-level number.
    region_sums = {region: 0.0 for region in REGIONS}
    for index in range(64):
        region_sums[_region_of_pixel(index)] += pixel_contributions.get(f"px_{index}", 0.0)

    # Pass 2: explain again with the raw image + region masks + the
    # region-aggregated contributions (replacing the 64-pixel-wide channel).
    result = fx.explain_one(
        flat,
        raw_object=sample_image,
        region_masks={name: _region_mask(name) for name in REGIONS},
        evidence={"contributions": region_sums},
    )

    print(f"native prediction: {int(torch.argmax(net(torch.as_tensor(X_test[:1]).unsqueeze(1)), dim=-1).item())}")
    print(f"FuzzyXAI prediction: {result.prediction.predictions}")
    print(f"{len(result.claims)} total claims (compare: a flat per-pixel scheme produced 1857 for a larger synthetic image)")
    print()
    print(result.summary())

    result.visualize(view="object_representation", output="/tmp/fuzzyxai_example_04.png")
    print("\nWrote /tmp/fuzzyxai_example_04.png (image with region boxes)")


if __name__ == "__main__":
    main()
