"""Freeze leakage-safe PAPILA patient-level outer folds before training."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from chapter6_medical_validation.ophthalmology.src.artifact_io import sha256_file, sha256_json, write_json_once
from chapter6_medical_validation.ophthalmology.src.datasets import configured_data_root, load_yaml

ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _patient_target(rows: list[dict[str, str]]) -> int:
    labels = {int(row["diagnosis"]) for row in rows}
    if not labels <= {0, 1}:
        raise ValueError("primary patient stratum contains a non-binary diagnosis")
    # Stratify patient groups by whether at least one eye has glaucoma.  Both
    # eyes remain inseparable for every split operation.
    return int(1 in labels)


def _patient_split(patient_ids: list[str], patient_targets: dict[str, int], fraction: float, seed: int) -> tuple[list[str], list[str]]:
    labels = [patient_targets[item] for item in patient_ids]
    try:
        train, validation = train_test_split(patient_ids, test_size=fraction, random_state=seed, stratify=labels)
    except ValueError:
        # A small fold can lack enough patients for two strata. This remains
        # deterministic and is recorded by the manifest, rather than leaking.
        train, validation = train_test_split(patient_ids, test_size=fraction, random_state=seed, stratify=None)
    return sorted(train), sorted(validation)


def freeze(data_root: Path, output: Path) -> dict[str, Any]:
    cfg = load_yaml(ROOT / "configs" / "dataset_papila.yaml")
    verified = data_root / "eyes" / "papila" / "verified"
    eye_rows = _rows(verified / "papila_eye_labels.csv")
    patients: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eye_rows:
        patients[row["patient_id"]].append(row)
    clean = {patient: rows for patient, rows in patients.items() if all(row["diagnosis"] in {"0", "1"} for row in rows)}
    suspect = {patient: rows for patient, rows in patients.items() if patient not in clean}
    if not clean or not suspect:
        raise ValueError("PAPILA protocol requires both clean binary and suspect-associated patient cohorts")
    targets = {patient: _patient_target(rows) for patient, rows in clean.items()}
    ids = np.asarray(sorted(clean)); labels = np.asarray([targets[item] for item in ids])
    splitter = StratifiedGroupKFold(n_splits=int(cfg["split"]["n_splits"]), shuffle=True, random_state=int(cfg["split"]["seed"]))
    fold_by_patient: dict[str, int] = {}
    for fold, (_train, test) in enumerate(splitter.split(ids, labels, groups=ids), start=1):
        for index in test:
            fold_by_patient[str(ids[index])] = fold
    if set(fold_by_patient) != set(clean):
        raise AssertionError("every clean patient must receive one outer fold")
    output.parent.mkdir(parents=True, exist_ok=True)
    folds: dict[str, Any] = {}
    for fold in range(1, int(cfg["split"]["n_splits"]) + 1):
        test_patients = sorted(patient for patient, value in fold_by_patient.items() if value == fold)
        pool = sorted(patient for patient, value in fold_by_patient.items() if value != fold)
        train_patients, val_patients = _patient_split(pool, targets, float(cfg["split"]["validation_fraction_of_training_patients"]), int(cfg["split"]["seed"]) + fold)
        sets = [set(train_patients), set(val_patients), set(test_patients)]
        if any(left & right for index, left in enumerate(sets) for right in sets[index + 1:]):
            raise AssertionError("patient leakage across train/validation/test")
        folds[str(fold)] = {
            "train_patient_ids": train_patients, "validation_patient_ids": val_patients, "test_patient_ids": test_patients,
            "counts": {"train_patients": len(train_patients), "validation_patients": len(val_patients), "test_patients": len(test_patients),
                       "train_eyes": sum(len(clean[item]) for item in train_patients), "validation_eyes": sum(len(clean[item]) for item in val_patients), "test_eyes": sum(len(clean[item]) for item in test_patients)},
            "patient_glaucoma_strata": {name: dict(Counter(targets[item] for item in values)) for name, values in {"train": train_patients, "validation": val_patients, "test": test_patients}.items()},
        }
    payload = {
        "schema_version": "1.0", "dataset_id": cfg["dataset_id"], "protocol": cfg["split"],
        "verified_manifest_sha256": sha256_file(verified / "papila_dataset_manifest.json"),
        "clean_binary_patient_count": len(clean), "clean_binary_eye_count": sum(len(rows) for rows in clean.values()),
        "suspect_associated_patient_count": len(suspect), "suspect_associated_eye_count": sum(len(rows) for rows in suspect.values()),
        "fold_by_patient": fold_by_patient, "folds": folds,
        "suspect_auxiliary_patients": sorted(suspect),
        "note": "All eyes of each patient occupy one outer fold. Suspect-associated patients are fully excluded from binary CV.",
    }
    payload["manifest_sha256"] = sha256_json(payload)
    write_json_once(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze PAPILA group-stratified CV split")
    parser.add_argument("--data-root")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = configured_data_root(args.data_root)
    output = args.output or root / "eyes" / "papila" / "verified" / "papila_cv_folds_seed2026.json"
    manifest = freeze(root, output)
    print(json.dumps({"output": str(output), "clean_binary_patients": manifest["clean_binary_patient_count"], "suspect_associated_patients": manifest["suspect_associated_patient_count"], "sha256": manifest["manifest_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
