from __future__ import annotations

import argparse
from pathlib import Path

from chapter6_medical_validation.ophthalmology.src.artifact_io import environment_manifest, write_json_once
from chapter6_medical_validation.ophthalmology.src.datasets import (
    assert_no_lesion_masks_in_classifier_inputs,
    build_dataset_manifest,
    configured_data_root,
    freeze_aptos_split,
    load_aptos_records,
    load_idrid_grading_split,
    load_yaml,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and freeze CH6 ophthalmology dataset manifests")
    parser.add_argument("--data-root")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "manifests")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    data_root = configured_data_root(args.data_root)
    aptos_cfg = load_yaml(ROOT / "configs" / "dataset_aptos.yaml")
    idrid_cfg = load_yaml(ROOT / "configs" / "dataset_idrid.yaml")
    aptos = load_aptos_records(data_root, aptos_cfg)
    assert_no_lesion_masks_in_classifier_inputs(record.image_path for record in aptos)
    idrid_train = load_idrid_grading_split(data_root, idrid_cfg, "train")
    idrid_test = load_idrid_grading_split(data_root, idrid_cfg, "test")
    assert_no_lesion_masks_in_classifier_inputs(record.image_path for record in idrid_train + idrid_test)
    if args.verify_only:
        print(f"APTOS verified: {len(aptos)}; IDRiD grading verified: {len(idrid_train) + len(idrid_test)}")
        return
    aptos_split = freeze_aptos_split(data_root, aptos_cfg, args.output / "split_aptos_seed2026.json")
    build_dataset_manifest(data_root, aptos, args.output / "aptos_inventory.json", source=aptos_cfg["source_url"])
    build_dataset_manifest(data_root, idrid_train + idrid_test, args.output / "idrid_grading_inventory.json", source=idrid_cfg["source_url"])
    write_json_once(args.output / "environment.json", environment_manifest())
    print(f"APTOS split frozen: {aptos_split['counts']}")
    print(f"IDRiD official grading split: train={len(idrid_train)}, test={len(idrid_test)}")


if __name__ == "__main__":
    main()
