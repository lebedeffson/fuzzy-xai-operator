from __future__ import annotations

import csv

import numpy as np
from PIL import Image

from chapter6_medical_validation.ophthalmology.src.datasets import (
    assert_no_lesion_masks_in_classifier_inputs,
    deterministic_aptos_split,
    load_aptos_records,
)


def _fixture(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for index in range(50):
        sample_id, label = f"eye_{index:03d}", index % 5
        Image.fromarray(np.full((8, 8, 3), 20 + index, dtype=np.uint8)).save(images / f"{sample_id}.png")
        rows.append({"id_code": sample_id, "diagnosis": label})
    with (tmp_path / "train.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id_code", "diagnosis"])
        writer.writeheader()
        writer.writerows(rows)
    config = {"dataset_id": "fixture", "relative_csv": "train.csv", "relative_images": "images", "image_extension": ".png", "expected_count": 50, "labels": [0, 1, 2, 3, 4], "id_column": "id_code", "label_column": "diagnosis", "split": {"seed": 2026, "counts": {"train": 30, "validation": 10, "internal_test": 10}}}
    return config


def test_aptos_loader_and_split_are_deterministic_and_disjoint(tmp_path):
    config = _fixture(tmp_path)
    records = load_aptos_records(tmp_path, config)
    first = deterministic_aptos_split(records, config)
    second = deterministic_aptos_split(records, config)
    assert {key: [item.sample_id for item in value] for key, value in first.items()} == {key: [item.sample_id for item in value] for key, value in second.items()}
    paths = [item.image_path for values in first.values() for item in values]
    assert len(paths) == len(set(paths)) == 50
    assert {item.label for item in records} == {0, 1, 2, 3, 4}


def test_lesion_masks_cannot_enter_classifier_input(tmp_path):
    bad = tmp_path / "segmentation" / "eye.png"
    try:
        assert_no_lesion_masks_in_classifier_inputs([bad])
    except ValueError as exc:
        assert "cannot enter classifier input" in str(exc)
    else:
        raise AssertionError("segmentation path was accepted")
