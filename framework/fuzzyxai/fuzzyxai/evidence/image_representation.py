from __future__ import annotations

import base64
import io
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

import numpy as np

from .contracts import ImageRegion, ImageRepresentationEvidence


def _as_image_array(raw_object: Any) -> np.ndarray | None:
    """Return a numpy pixel array for a raw object, or ``None`` if it isn't image-shaped.

    Recognizes a numpy array with 2 dims (grayscale HxW) or 3 dims (HxWxC),
    and anything exposing PIL's ``.size``/``.mode`` (converted via
    ``np.asarray`` without a hard Pillow dependency — matplotlib already
    depends on it transitively, but this module never imports PIL directly).
    Never guesses at 1D arrays (those are ordinary tabular feature vectors,
    not images) or higher-dimensional arrays (batches are out of scope here
    — this is a single-object representation, like ``raw_object`` for text).
    """

    if isinstance(raw_object, np.ndarray):
        array = raw_object
    elif hasattr(raw_object, "size") and hasattr(raw_object, "mode"):
        array = np.asarray(raw_object)
    else:
        return None
    if array.ndim not in (2, 3):
        return None
    return array


def is_image_like(raw_object: Any) -> bool:
    """Cheap type check used by the canonical runtime to route raw_object dispatch."""

    return _as_image_array(raw_object) is not None


def find_image_regions(
    raw_object: Any,
    feature_contributions: Mapping[str, float] | None,
    *,
    object_id: str,
    region_masks: Mapping[str, Sequence[Sequence[bool]]] | None = None,
) -> ImageRepresentationEvidence:
    """Build a typed image representation: real dimensions, real region geometry, no invented attribution.

    ``region_masks`` is optional and caller-supplied — e.g. from a
    segmentation model or manual annotation. Each mask must match the
    image's own (height, width); a mismatched mask is dropped with an
    explicit limitation, never silently resized or ignored without a trace.
    Without any region_masks, ``regions`` is empty and a limitation
    discloses that no region/attribution evidence was supplied — FuzzyXAI
    has no built-in per-pixel attribution method (no Grad-CAM/Integrated
    Gradients adapter), so it never fabricates one.

    A region's ``direction``/``contribution`` are populated only when
    ``feature_contributions`` has a matching key for that region's name;
    otherwise "unknown"/``None``, matching the existing tabular-fallback
    honesty convention.
    """

    array = _as_image_array(raw_object)
    if array is None:
        raise TypeError("raw_object is not image-shaped (expected a 2D or 3D array-like object)")
    height, width = array.shape[0], array.shape[1]
    channels = array.shape[2] if array.ndim == 3 else 1

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    normalized = array.astype("float64")
    value_range = float(normalized.max() - normalized.min())
    if value_range > 0 and (normalized.max() > 1.0 or normalized.min() < 0.0):
        # imsave expects either [0, 1] floats or [0, 255] ints; anything
        # outside that (e.g. a raw sensor array) is min-max normalized only
        # for the *rendered artifact* — the region masks below still operate
        # on the original array's own coordinate space, so no measurement is
        # affected by this display-only rescaling.
        normalized = (normalized - normalized.min()) / value_range
    plt.imsave(buffer, normalized.squeeze() if channels == 1 else normalized, format="png", cmap="gray" if channels == 1 else None)
    png_bytes = buffer.getvalue()
    artifact_hash = sha256(png_bytes).hexdigest()
    image_base64 = base64.b64encode(png_bytes).decode("ascii")

    regions: list[ImageRegion] = []
    limitations: list[str] = []
    contributions = feature_contributions or {}
    if not region_masks:
        limitations.append("no region masks were supplied; per-pixel attribution was not measured, not omitted by mistake")
    else:
        for name, mask in region_masks.items():
            mask_array = np.asarray(mask, dtype=bool)
            if mask_array.shape != (height, width):
                limitations.append(f"region '{name}' mask shape {mask_array.shape} does not match image shape ({height}, {width}); dropped")
                continue
            if not mask_array.any():
                limitations.append(f"region '{name}' mask is empty (no True pixels); dropped")
                continue
            rows, cols = np.nonzero(mask_array)
            contribution = contributions.get(name)
            numeric_contribution = float(contribution) if isinstance(contribution, (int, float)) else None
            direction: str = "unknown" if numeric_contribution is None else ("supports" if numeric_contribution >= 0 else "contradicts")
            regions.append(
                ImageRegion(
                    name=str(name),
                    pixel_count=int(mask_array.sum()),
                    bounding_box=(int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())),
                    direction=direction,  # type: ignore[arg-type]
                    contribution=numeric_contribution,
                )
            )

    return ImageRepresentationEvidence(
        object_id=str(object_id),
        width=int(width),
        height=int(height),
        channels=int(channels),
        artifact_sha256=artifact_hash,
        image_png_base64=image_base64,
        regions=tuple(regions),
        limitations=tuple(limitations),
    )
