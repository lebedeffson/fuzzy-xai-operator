"""P3: first-class image object_representation.

Extends the existing text/tabular ``object_representation`` discriminated
union with an ``"image"`` modality, using the same evidence-first rules:
real dimensions measured from the array, region geometry measured directly
from caller-supplied boolean masks (never inferred/fabricated), no binary
image bytes in JSON by default, and ``compare_region_masks`` stays a
pixel-mask IoU comparator (not an image renderer or RGB comparison tool).
"""

from __future__ import annotations

import base64

import numpy as np
import pytest
from fuzzyxai import FuzzyXAI
from fuzzyxai.evidence import (
    ExplanationEvidence,
    build_explanation_claims,
    build_explanation_graph,
    compose_human_explanation,
    find_image_regions,
    is_image_like,
)
from sklearn.linear_model import LogisticRegression


def _model_and_image():
    rng = np.random.default_rng(0)
    X = rng.random((40, 30 * 20 * 3))
    y = (X[:, 0] > 0.5).astype(int)
    model = LogisticRegression(max_iter=2000).fit(X, y)
    image = rng.random((20, 30, 3))
    return model, image


def test_is_image_like_accepts_2d_and_3d_arrays_rejects_tabular_vector() -> None:
    assert is_image_like(np.zeros((20, 30)))
    assert is_image_like(np.zeros((20, 30, 3)))
    assert not is_image_like([1.0, 2.0, 3.0])
    assert not is_image_like(np.zeros(10))  # 1D — an ordinary feature vector, not an image


def test_raw_image_accepted_and_dimensions_are_correct() -> None:
    model, image = _model_and_image()
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    result = fx.explain_one(image.flatten(), object_id="img0", raw_object=image)
    repr_ = result.object_representation
    assert repr_["modality"] == "image"
    assert repr_["image_height"] == 20
    assert repr_["image_width"] == 30
    assert repr_["image_channels"] == 3


def test_wrong_raw_object_type_handled_honestly_not_silently() -> None:
    """A raw_object that is neither text nor image falls back to the tabular
    view with an explicit missing-channel disclosure, never a fabricated image."""

    model, image = _model_and_image()
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    result = fx.explain_one(image.flatten(), object_id="img0", raw_object={"not": "an image"})
    assert "text_highlight_unsupported_raw_object_type" in result.view_model.trace["missing_evidence"]
    assert result.object_representation["modality"] == "tabular"


def test_no_region_masks_means_empty_regions_with_honest_limitation() -> None:
    model, image = _model_and_image()
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    result = fx.explain_one(image.flatten(), object_id="img0", raw_object=image)
    repr_ = result.object_representation
    assert repr_["image_regions"] == ()
    assert any("no region masks" in item for item in repr_["limitations"])


def test_regions_align_with_supplied_masks_and_no_region_is_invented() -> None:
    model, image = _model_and_image()
    mask = np.zeros((20, 30), dtype=bool)
    mask[2:8, 5:15] = True  # 6 rows x 10 cols = 60 pixels
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    result = fx.explain_one(image.flatten(), object_id="img0", raw_object=image, region_masks={"lesion": mask})
    regions = result.object_representation["image_regions"]
    assert len(regions) == 1
    region = regions[0]
    assert region["name"] == "lesion"
    assert region["pixel_count"] == 60
    assert tuple(region["bounding_box"]) == (2, 7, 5, 14)


def test_mismatched_mask_shape_is_dropped_with_a_traceable_limitation() -> None:
    model, image = _model_and_image()
    bad_mask = np.zeros((5, 5), dtype=bool)
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    result = fx.explain_one(image.flatten(), object_id="img0", raw_object=image, region_masks={"bad": bad_mask})
    repr_ = result.object_representation
    assert repr_["image_regions"] == ()
    assert any("does not match image shape" in item for item in repr_["limitations"])


def test_json_export_does_not_contain_raw_image_bytes_by_default() -> None:
    model, image = _model_and_image()
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    result = fx.explain_one(image.flatten(), object_id="img0", raw_object=image)
    for detail in ("compact", "standard", "audit"):
        payload = result.to_dict(detail=detail, include_raw=False)
        blob = str(payload)
        assert "image_png_base64" not in blob or "omitted by default" in blob
        # sha256 (an integrity reference, not content) is retained even when redacted.
        assert result.object_representation["image_artifact_sha256"] in blob or detail == "compact"


def test_include_raw_true_restores_decodable_png_bytes() -> None:
    model, image = _model_and_image()
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    result = fx.explain_one(image.flatten(), object_id="img0", raw_object=image)
    payload = result.to_dict(detail="audit", include_raw=True)
    encoded = payload["visual_spec"]["object_representation"]["image_png_base64"]
    png_bytes = base64.b64decode(encoded)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"  # real PNG magic bytes, not a placeholder


def test_matplotlib_renderer_works_headless_for_image_modality() -> None:
    model, image = _model_and_image()
    mask = np.zeros((20, 30), dtype=bool)
    mask[2:8, 5:15] = True
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    result = fx.explain_one(image.flatten(), object_id="img0", raw_object=image, region_masks={"lesion": mask})
    figure = result.visualize(view="object_representation", backend="matplotlib")
    assert figure is not None


def test_plotly_renderer_works_for_image_modality() -> None:
    model, image = _model_and_image()
    mask = np.zeros((20, 30), dtype=bool)
    mask[2:8, 5:15] = True
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    result = fx.explain_one(image.flatten(), object_id="img0", raw_object=image, region_masks={"lesion": mask})
    figure = result.visualize(view="object_representation", backend="plotly")
    assert figure is not None


def test_measured_region_contribution_produces_supporting_or_contradicting_claim() -> None:
    image = np.random.default_rng(1).random((20, 30, 3))
    mask = np.zeros((20, 30), dtype=bool)
    mask[2:8, 5:15] = True
    image_evidence = find_image_regions(image, {"lesion": 0.85}, object_id="img0", region_masks={"lesion": mask})
    evidence = ExplanationEvidence(image_representations=[image_evidence])
    prediction = {"predictions": [1], "score": 0.9}
    claims = build_explanation_claims(evidence, prediction=prediction, diagnostics=[], action="accept")
    region_claims = [claim for claim in claims if claim.claim_type == "image_region"]
    assert len(region_claims) == 1
    assert region_claims[0].effect == "favorable"
    assert region_claims[0].metric_value == pytest.approx(0.85)


def test_unmeasured_region_contribution_is_not_fabricated() -> None:
    """A region name with no matching entry in the model's contribution
    mapping must not silently be given a fake direction/magnitude."""

    image = np.random.default_rng(2).random((20, 30, 3))
    mask = np.zeros((20, 30), dtype=bool)
    mask[2:8, 5:15] = True
    image_evidence = find_image_regions(image, {}, object_id="img0", region_masks={"lesion": mask})
    evidence = ExplanationEvidence(image_representations=[image_evidence])
    prediction = {"predictions": [1], "score": 0.9}
    claims = build_explanation_claims(evidence, prediction=prediction, diagnostics=[], action="accept")
    region_claim = next(claim for claim in claims if claim.claim_type == "image_region")
    assert region_claim.effect == "neutral"
    assert region_claim.strength is None
    assert region_claim.metric_name == "region_pixel_count"

    graph = build_explanation_graph(evidence, prediction=prediction, diagnostics=[], action="accept", claims=claims)
    human = compose_human_explanation(claims, graph, action="accept", audience="researcher", evidence=evidence)
    statement = next(item for item in human.details.supports if item.title == "Область lesion")
    assert "не измерен" in statement.explanation
    assert "60" not in statement.explanation.split("вклад")[0]  # pixel count must not be presented as a contribution number


def test_compare_region_masks_is_not_described_as_an_image_renderer() -> None:
    """Regression guard for the documented distinction: compare_region_masks
    is a pixel-mask IoU comparator, not an RGB/image overlay generator."""

    from fuzzyxai.evidence import compare_region_masks

    query = np.zeros((10, 10), dtype=bool)
    query[2:5, 2:5] = True
    reference = np.zeros((10, 10), dtype=bool)
    reference[2:5, 2:5] = True
    result = compare_region_masks(query, reference, query_object_id="q", reference_object_id="r")
    assert result.similarity_method == "binary_mask_intersection_over_union"
    assert result.compared_representation == "segmentation masks (pixels inside the selected region)"
