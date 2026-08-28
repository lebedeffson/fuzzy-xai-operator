from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import yaml

from chapter6_medical_validation.brain_allen.src.data import descendants, load_structures, load_volumes, patch_hash
from chapter6_medical_validation.shared.hashing import sha256_file, sha256_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data_root = os.environ.get("FUZZYXAI_CH6_DATA_ROOT")
    if not data_root: raise FileNotFoundError("FUZZYXAI_CH6_DATA_ROOT is not set")
    atlas_root = Path(data_root) / "brain" / "allen_ccf_25um"
    config = yaml.safe_load((ROOT / "configs" / "dataset_allen.yaml").read_text())
    nissl, annotation, nissl_header, annotation_header = load_volumes(atlas_root)
    source_shape = list(nissl.shape)
    structures = load_structures(atlas_root / "structure_graph.json")
    hpf_ids, gray_ids = descendants(structures, "HPF"), descendants(structures, "grey")
    axis = int(config["coronal_axis"])
    nissl = np.moveaxis(nissl, axis, 0)
    annotation = np.moveaxis(annotation, axis, 0)
    hpf = np.isin(annotation, list(hpf_ids)); gray = np.isin(annotation, list(gray_ids))
    size, stride, block_size = int(config["patch_size"]), int(config["stride"]), int(config["section_block_size"])
    candidates = []
    for section in range(nissl.shape[0]):
        for row in range(0, nissl.shape[1] - size + 1, stride):
            for col in range(0, nissl.shape[2] - size + 1, stride):
                image = nissl[section, row : row + size, col : col + size]
                hpf_fraction = float(hpf[section, row : row + size, col : col + size].mean())
                gray_fraction = float(gray[section, row : row + size, col : col + size].mean())
                label = 1 if hpf_fraction >= float(config["positive_min_hpf_fraction"]) else (0 if hpf_fraction <= float(config["negative_max_hpf_fraction"]) and gray_fraction >= float(config["min_gray_tissue_fraction"]) else None)
                if label is not None:
                    candidates.append({"section": section, "block": section // block_size, "row": row, "col": col, "label": label, "hpf_fraction": hpf_fraction, "gray_fraction": gray_fraction, "hash": patch_hash(image)})
    rng = np.random.default_rng(int(config["split_seed"])); by_class = {label: [item for item in candidates if item["label"] == label] for label in (0, 1)}
    per_class = min(len(by_class[0]), len(by_class[1]), int(config["max_patches_per_class"]))
    selected = []
    for label in (0, 1):
        indexes = rng.choice(len(by_class[label]), per_class, replace=False); selected.extend(by_class[label][int(index)] for index in indexes)
    block_labels = {block: {item["label"] for item in selected if item["block"] == block} for block in {item["block"] for item in selected}}
    positive_blocks = sorted(block for block, labels in block_labels.items() if 1 in labels)
    negative_only_blocks = sorted(block for block, labels in block_labels.items() if labels == {0})
    rng.shuffle(positive_blocks); rng.shuffle(negative_only_blocks)
    def assigned(values):
        if len(values) < 3: raise ValueError("at least three independent section blocks are required per stratum")
        n_validation = max(1, round(len(values) * 0.15)); n_test = max(1, round(len(values) * 0.15)); n_train = len(values) - n_validation - n_test
        return {block: ("train" if index < n_train else "validation" if index < n_train + n_validation else "test") for index, block in enumerate(values)}
    assignment = {**assigned(positive_blocks), **assigned(negative_only_blocks)}
    boundary = int(config["boundary_exclusion_slices"])
    kept = []
    for item in selected:
        split = assignment[item["block"]]; adjacent = {assignment.get(item["block"] - 1), assignment.get(item["block"] + 1)} - {None, split}
        offset = item["section"] % block_size
        if adjacent and (offset < boundary or offset >= block_size - boundary): continue
        kept.append({**item, "split": split})
    hashes: dict[str, str] = {}
    for item in kept:
        previous = hashes.setdefault(item["hash"], item["split"])
        if previous != item["split"]: raise ValueError("near-identical exact patch hash crosses split")
    for split in ("train", "validation", "test"):
        if {item["label"] for item in kept if item["split"] == split} != {0, 1}:
            raise ValueError(f"section-block split {split} does not contain both classes")
    prepared = atlas_root / "prepared"; prepared.mkdir(exist_ok=True)
    arrays = np.empty((len(kept), size, size), dtype=nissl.dtype); masks = np.empty((len(kept), size, size), dtype=np.uint8)
    for index, item in enumerate(kept):
        arrays[index] = nissl[item["section"], item["row"] : item["row"] + size, item["col"] : item["col"] + size]
        masks[index] = hpf[item["section"], item["row"] : item["row"] + size, item["col"] : item["col"] + size]
    np.save(prepared / "patches.npy", arrays); np.save(prepared / "hpf_masks.npy", masks)
    (prepared / "patches.json").write_text(json.dumps(kept, indent=2) + "\n")
    counts = {split: sum(item["split"] == split for item in kept) for split in ("train", "validation", "test")}
    manifest = {"dataset_name": config["dataset_name"], "dataset_version": config["dataset_version"], "resolution_um": 25, "source_volume_shape": source_shape, "coronal_working_shape": list(nissl.shape), "coronal_axis_source": axis, "nissl_header_space": str(nissl_header.get("space")), "annotation_header_space": str(annotation_header.get("space")), "hpf_structure_count": len(hpf_ids), "gray_structure_count": len(gray_ids), "patch_counts": counts, "class_counts": {str(label): sum(item["label"] == label for item in kept) for label in (0, 1)}, "source_sha256": {name: sha256_file(atlas_root / name) for name in ("ara_nissl_25.nrrd", "annotation_25.nrrd", "structure_graph.json")}, "prepared_sha256": {name: sha256_file(prepared / name) for name in ("patches.npy", "hpf_masks.npy", "patches.json")}, "limitations": ["single-atlas section-block generalization, not subject-level generalization"]}
    manifest["manifest_sha256"] = sha256_json(manifest); (prepared / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
