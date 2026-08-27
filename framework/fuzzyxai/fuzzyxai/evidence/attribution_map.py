from __future__ import annotations

import base64
import io
from collections.abc import Mapping
from typing import Any

import numpy as np

from .contracts import AttributionMapEvidence

# Adapter channel names known to carry a full per-pixel/per-voxel
# attribution tensor (same spatial shape as the input), in priority order.
# New adapters can add their own channel name here without touching runtime.py.
KNOWN_ATTRIBUTION_CHANNELS: tuple[str, ...] = ("integrated_gradients", "saliency_map", "grad_cam", "attribution_map")

# P17: Russian labels for the renderer's caption text — never leave the
# internal English channel name as the only reader-facing label.
_METHOD_LABELS_RU = {
    "integrated_gradients": "интегрированные градиенты",
    "saliency_map": "карта значимости",
    "grad_cam": "Grad-CAM",
    "attribution_map": "карта атрибуции",
}


def _aggregate_channels(array: np.ndarray, *, image_channels: int) -> tuple[np.ndarray, str]:
    """Collapse a (possibly channel-first) attribution tensor to one 2D heatmap.

    PyTorch convention is channel-first (C, H, W); the raw image is
    channel-last (H, W, C) or plain (H, W). Aggregation sums across whatever
    axis matches the image's own channel count so the heatmap aligns pixel
    for pixel with the rendered base image — never a guessed transpose.
    """

    values = np.asarray(array, dtype=float)
    if values.ndim == 2:
        return values, "single_channel"
    if values.ndim == 3:
        if values.shape[0] == image_channels and values.shape[0] != values.shape[-1]:
            return values.sum(axis=0), "sum_over_channels_first_axis"
        if values.shape[-1] == image_channels:
            return values.sum(axis=-1), "sum_over_channels_last_axis"
        return values.reshape(values.shape[-2], values.shape[-1]) if values.shape[0] == 1 else values.sum(axis=0), "sum_over_leading_axis"
    if values.ndim == 4 and values.shape[0] == 1:
        return _aggregate_channels(values[0], image_channels=image_channels)
    raise ValueError(f"attribution array has unsupported shape {values.shape} for image overlay")


def _label_connected_components(mask: np.ndarray) -> np.ndarray:
    """4-connectivity connected-component labeling on a boolean grid, pure
    numpy (no new dependency) — a plain iterative flood fill. Returns an
    int array where 0 = background and each connected True-region gets its
    own positive label."""

    labels = np.zeros(mask.shape, dtype=int)
    next_label = 1
    height, width = mask.shape
    for start_row in range(height):
        for start_col in range(width):
            if not mask[start_row, start_col] or labels[start_row, start_col] != 0:
                continue
            stack = [(start_row, start_col)]
            labels[start_row, start_col] = next_label
            while stack:
                row, col = stack.pop()
                for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    n_row, n_col = row + d_row, col + d_col
                    if 0 <= n_row < height and 0 <= n_col < width and mask[n_row, n_col] and labels[n_row, n_col] == 0:
                        labels[n_row, n_col] = next_label
                        stack.append((n_row, n_col))
            next_label += 1
    return labels


def find_attribution_regions(
    attribution_array: Any,
    *,
    image_shape: tuple[int, int],
    image_channels: int = 1,
    magnitude_percentile: float = 70.0,
    min_pixels: int = 3,
) -> dict[str, np.ndarray]:
    """Derive region masks algorithmically from the real attribution map —
    connected components of pixels whose |attribution| exceeds a percentile
    threshold — instead of an arbitrary fixed grid (e.g. four quadrants).

    Returns a dict of region name -> boolean mask, empty when nothing
    exceeds the threshold. The source of each region is exactly this
    threshold + connected-components procedure — no other geometry is
    invented.
    """

    heatmap, _ = _aggregate_channels(np.asarray(attribution_array), image_channels=image_channels)
    if heatmap.shape != image_shape:
        raise ValueError(f"aggregated attribution shape {heatmap.shape} does not match image_shape {image_shape}")
    magnitude = np.abs(heatmap)
    if not np.any(magnitude > 0):
        return {}
    threshold = float(np.percentile(magnitude[magnitude > 0], magnitude_percentile))
    mask = magnitude >= threshold
    labels = _label_connected_components(mask)
    regions: dict[str, np.ndarray] = {}
    for label in range(1, int(labels.max()) + 1):
        region_mask = labels == label
        if int(region_mask.sum()) < min_pixels:
            continue
        regions[f"attribution_region_{label}"] = region_mask
    return regions


def build_attribution_map(
    raw_object: Any,
    attribution_array: Any,
    *,
    object_id: str,
    method: str,
    target: str | None = None,
    baseline: str = "unspecified baseline",
    completeness_error: float | None = None,
    completeness: Mapping[str, Any] | None = None,
    source_refs: tuple[str, ...] = (),
) -> AttributionMapEvidence:
    """Preserve a full attribution tensor as first-class evidence and render
    it as a real semi-transparent overlay — never four arbitrary quadrants.

    ``attribution_array`` must already be a real, measured tensor from the
    adapter (e.g. TorchAdapter's Integrated Gradients) — this function does
    not compute attribution itself, only preserves and renders what was
    already measured.
    """

    from .image_representation import _as_image_array

    image = _as_image_array(raw_object)
    if image is None:
        raise TypeError("raw_object is not image-shaped (expected a 2D or 3D array-like object)")
    image_channels = image.shape[2] if image.ndim == 3 else 1
    heatmap, channel_aggregation = _aggregate_channels(np.asarray(attribution_array), image_channels=image_channels)
    if heatmap.shape != image.shape[:2]:
        raise ValueError(f"aggregated attribution shape {heatmap.shape} does not match image shape {image.shape[:2]}")

    positive_sum = float(heatmap[heatmap > 0].sum()) if np.any(heatmap > 0) else 0.0
    negative_sum = float(heatmap[heatmap < 0].sum()) if np.any(heatmap < 0) else 0.0
    min_value = float(heatmap.min())
    max_value = float(heatmap.max())

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    base = image.astype("float64")
    base_range = float(base.max() - base.min())
    if base_range > 0 and (base.max() > 1.0 or base.min() < 0.0):
        base = (base - base.min()) / base_range
    base_gray = base if base.ndim == 2 else base.mean(axis=-1)

    # P17: no title baked into the image itself (the caption belongs in the
    # surrounding document/report, not the picture — see P16 section 22),
    # a Russian legend for the method, and enough room for the colorbar
    # label so it isn't cramped against the plot.
    scale = max(abs(min_value), abs(max_value)) or 1.0
    method_label = _METHOD_LABELS_RU.get(method, method)
    fig, (ax_image, ax_bar) = plt.subplots(1, 2, figsize=(8.4, 4.6), gridspec_kw={"width_ratios": [10, 1.2]})
    ax_image.imshow(base_gray, cmap="gray")
    overlay = ax_image.imshow(heatmap, cmap="bwr", vmin=-scale, vmax=scale, alpha=0.5)
    ax_image.axis("off")
    ax_image.text(0.5, -0.05, f"метод: {method_label}", transform=ax_image.transAxes, ha="center", va="top", fontsize=9)
    colorbar = fig.colorbar(overlay, cax=ax_bar)
    colorbar.set_label("вклад пикселя", rotation=90, labelpad=8, fontsize=9)
    fig.text(0.98, 0.02, "синий = противоречит, красный = поддерживает", ha="right", fontsize=8, color="#444444")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    attribution_png_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    return AttributionMapEvidence(
        object_id=str(object_id),
        method=method,
        target=target,
        baseline=baseline,
        shape=tuple(int(dim) for dim in np.asarray(attribution_array).shape),
        channel_aggregation=channel_aggregation,
        min_value=min_value,
        max_value=max_value,
        positive_sum=positive_sum,
        negative_sum=negative_sum,
        attribution_array=np.asarray(attribution_array).tolist(),
        attribution_png_base64=attribution_png_base64,
        completeness_error=completeness_error,
        completeness=dict(completeness or {"status": "not_evaluated", "reason": "adapter did not provide a same-output-space completeness calculation"}),
        source_refs=source_refs,
        limitations=("Attribution reflects this model's own gradient behavior, not domain causality.",),
    )
