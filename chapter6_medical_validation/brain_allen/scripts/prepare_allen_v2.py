"""Prepare the pre-registered, section-block-safe Allen CCF v2 cohort.

This script never reads v1 model metrics.  It writes a separate prepared
directory and manifest, leaving the 95-patch v1 pilot unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt

from chapter6_medical_validation.brain_allen.src.data import descendants, load_structures, load_volumes, patch_hash
from chapter6_medical_validation.shared.hashing import sha256_file, sha256_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "dataset_allen_v2_confirmatory.yaml"


def _assign_blocks(blocks: list[int], *, seed: int, fractions: dict[str, float]) -> dict[int, str]:
    """Assign whole section blocks before sampling any patches."""

    if len(blocks) < 20:
        raise ValueError(f"confirmatory protocol needs >=20 HPF-containing blocks, got {len(blocks)}")
    rng = np.random.default_rng(seed)
    ordered = list(blocks)
    rng.shuffle(ordered)
    n_test = max(10, round(len(ordered) * float(fractions["test"])))
    n_validation = max(5, round(len(ordered) * float(fractions["validation"])))
    n_train = len(ordered) - n_validation - n_test
    if n_train < 1:
        raise ValueError("section-block allocation leaves no training blocks")
    return {
        block: "train" if index < n_train else "validation" if index < n_train + n_validation else "test"
        for index, block in enumerate(ordered)
    }


def main() -> None:
    data_root = os.environ.get("FUZZYXAI_CH6_DATA_ROOT")
    if not data_root:
        raise FileNotFoundError("FUZZYXAI_CH6_DATA_ROOT is not set")
    atlas_root = Path(data_root) / "brain" / "allen_ccf_25um"
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    nissl, annotation, nissl_header, annotation_header = load_volumes(atlas_root)
    source_shape = list(nissl.shape)
    structures = load_structures(atlas_root / str(config["ontology_file"]))
    hpf_ids = descendants(structures, str(config["positive_structure_acronym"]))
    gray_ids = descendants(structures, str(config["gray_matter_acronym"]))
    axis = int(config["coronal_axis"])
    nissl, annotation = np.moveaxis(nissl, axis, 0), np.moveaxis(annotation, axis, 0)
    hpf, gray = np.isin(annotation, list(hpf_ids)), np.isin(annotation, list(gray_ids))
    size, stride, block_size = (int(config[name]) for name in ("patch_size", "stride", "section_block_size"))
    candidates: list[dict[str, object]] = []
    for section in range(nissl.shape[0]):
        # The distance is purely a registered anatomical hard-negative sampler.
        near_hpf = distance_transform_edt(~hpf[section]) <= float(config["hard_negative_max_distance_voxels"])
        for row in range(0, nissl.shape[1] - size + 1, stride):
            for col in range(0, nissl.shape[2] - size + 1, stride):
                hpf_fraction = float(hpf[section, row : row + size, col : col + size].mean())
                gray_fraction = float(gray[section, row : row + size, col : col + size].mean())
                label: int | None
                if hpf_fraction >= float(config["positive_min_hpf_fraction"]) and gray_fraction >= float(config["positive_min_gray_tissue_fraction"]):
                    label = 1
                elif (
                    hpf_fraction <= float(config["negative_max_hpf_fraction"])
                    and gray_fraction >= float(config["negative_min_gray_tissue_fraction"])
                    and bool(near_hpf[row : row + size, col : col + size].any())
                ):
                    label = 0
                else:
                    label = None
                if label is not None:
                    patch = nissl[section, row : row + size, col : col + size]
                    candidates.append({"section": section, "block": section // block_size, "row": row, "col": col, "label": label, "hpf_fraction": hpf_fraction, "gray_fraction": gray_fraction, "hash": patch_hash(patch)})
    hpf_blocks = sorted({int(item["block"]) for item in candidates if item["label"] == 1})
    assignment = _assign_blocks(hpf_blocks, seed=int(config["split_seed"]), fractions=dict(config["split_fractions"]))
    # Blocks containing only hard negatives are assigned deterministically to
    # training; no test/validation test is inflated by those blocks.
    kept_candidates = [{**item, "split": assignment.get(int(item["block"]), "train")} for item in candidates]
    rng = np.random.default_rng(int(config["split_seed"]))
    per_class = min(*(sum(item["label"] == label for item in kept_candidates) for label in (0, 1)), int(config["max_patches_per_class"]))
    selected: list[dict[str, object]] = []
    for label in (0, 1):
        rows = [item for item in kept_candidates if item["label"] == label]
        indexes = rng.choice(len(rows), per_class, replace=False)
        selected.extend(rows[int(index)] for index in indexes)
    boundary = int(config["boundary_exclusion_slices"])
    kept: list[dict[str, object]] = []
    for item in selected:
        split = str(item["split"])
        adjacent = {assignment.get(int(item["block"]) - 1), assignment.get(int(item["block"]) + 1)} - {None, split}
        offset = int(item["section"]) % block_size
        if adjacent and (offset < boundary or offset >= block_size - boundary):
            continue
        kept.append(item)
    hashes: dict[str, str] = {}
    for item in kept:
        previous = hashes.setdefault(str(item["hash"]), str(item["split"]))
        if previous != item["split"]:
            raise ValueError("exact patch hash crosses a section-block split")
    for split in ("train", "validation", "test"):
        labels = {int(item["label"]) for item in kept if item["split"] == split}
        if labels != {0, 1}:
            raise ValueError(f"v2 split {split} does not contain both classes: {labels}")
    test_blocks = {int(item["block"]) for item in kept if item["split"] == "test"}
    if len(test_blocks) < 10:
        raise ValueError(f"v2 confirmatory test has only {len(test_blocks)} independent section blocks")
    prepared = atlas_root / "prepared_v2_confirmatory"
    if prepared.exists():
        raise FileExistsError(f"refusing to overwrite frozen v2 preparation: {prepared}")
    prepared.mkdir()
    arrays = np.empty((len(kept), size, size), dtype=nissl.dtype)
    masks = np.empty((len(kept), size, size), dtype=np.uint8)
    for index, item in enumerate(kept):
        section, row, col = (int(item[name]) for name in ("section", "row", "col"))
        arrays[index] = nissl[section, row : row + size, col : col + size]
        masks[index] = hpf[section, row : row + size, col : col + size]
    np.save(prepared / "patches.npy", arrays)
    np.save(prepared / "hpf_masks.npy", masks)
    (prepared / "patches.json").write_text(json.dumps(kept, indent=2) + "\n", encoding="utf-8")
    counts = {split: sum(item["split"] == split for item in kept) for split in ("train", "validation", "test")}
    manifest = {
        "protocol_id": config["protocol_id"],
        "config_sha256": sha256_file(CONFIG_PATH),
        "dataset_name": config["dataset_name"],
        "dataset_version": config["dataset_version"],
        "resolution_um": config["resolution_um"],
        "source_volume_shape": source_shape,
        "coronal_working_shape": list(nissl.shape),
        "coronal_axis_source": axis,
        "nissl_header_space": str(nissl_header.get("space")),
        "annotation_header_space": str(annotation_header.get("space")),
        "sampling_protocol": {key: config[key] for key in ("patch_size", "stride", "positive_min_hpf_fraction", "positive_min_gray_tissue_fraction", "negative_max_hpf_fraction", "negative_min_gray_tissue_fraction", "hard_negative_max_distance_voxels", "section_block_size", "boundary_exclusion_slices")},
        "patch_counts": counts,
        "class_counts": {str(label): sum(item["label"] == label for item in kept) for label in (0, 1)},
        "independent_section_blocks": {split: len({int(item["block"]) for item in kept if item["split"] == split}) for split in ("train", "validation", "test")},
        "source_sha256": {name: sha256_file(atlas_root / name) for name in (str(config["nissl_file"]), str(config["annotation_file"]), str(config["ontology_file"]))},
        "prepared_sha256": {name: sha256_file(prepared / name) for name in ("patches.npy", "hpf_masks.npy", "patches.json")},
        "limitations": config["limitations"],
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    (prepared / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
