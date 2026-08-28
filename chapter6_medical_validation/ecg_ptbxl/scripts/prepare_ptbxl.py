from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from chapter6_medical_validation.ecg_ptbxl.src.data import assert_patient_disjoint, derive_primary_records, load_metadata, load_waveform, record_dict
from chapter6_medical_validation.ecg_ptbxl.src.preprocessing import fit_lead_statistics
from chapter6_medical_validation.shared.hashing import sha256_file, sha256_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and prepare official PTB-XL binary cohort")
    parser.add_argument("--verify-sha", action="store_true")
    args = parser.parse_args()
    data_root = os.environ.get("FUZZYXAI_CH6_DATA_ROOT")
    if not data_root:
        raise FileNotFoundError("FUZZYXAI_CH6_DATA_ROOT is not set")
    root = Path(data_root) / "ecg" / "ptb-xl-1.0.3"
    database, statements = load_metadata(root)
    records = derive_primary_records(database, statements)
    assert_patient_disjoint(records)
    included = [record for record in records if record.included]
    prepared = root / "prepared"
    prepared.mkdir(exist_ok=True)
    pd.DataFrame([record_dict(record) for record in records]).to_csv(prepared / "label_construction.csv", index=False)
    signals = np.lib.format.open_memmap(prepared / "signals.npy", mode="w+", dtype=np.float32, shape=(len(included), 12, 1000))
    labels = np.empty(len(included), dtype=np.int64); folds = np.empty(len(included), dtype=np.int8); ids = np.empty(len(included), dtype=np.int64)
    for index, record in enumerate(included):
        signals[index] = load_waveform(root, record); labels[index] = int(record.label); folds[index] = record.fold; ids[index] = record.ecg_id
        if (index + 1) % 1000 == 0:
            print(f"loaded {index + 1}/{len(included)}", flush=True)
    signals.flush(); np.save(prepared / "labels.npy", labels); np.save(prepared / "folds.npy", folds); np.save(prepared / "ecg_ids.npy", ids)
    train_indices = np.flatnonzero(folds <= 8)
    stats = fit_lead_statistics(signals[index] for index in train_indices)
    (prepared / "normalization.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    counts = {"total_metadata": len(records), "included_primary": len(included), "excluded": len(records) - len(included), "train": int((folds <= 8).sum()), "validation": int((folds == 9).sum()), "test": int((folds == 10).sum()), "normal": int((labels == 0).sum()), "abnormal": int((labels == 1).sum())}
    manifest = {"dataset_name": "PTB-XL", "dataset_version": "1.0.3", "official_source": "https://physionet.org/files/ptb-xl/1.0.3/", "doi": "10.13026/kfzx-aw45", "license": "PhysioNet Credentialed Health Data License 1.5.0 as shipped in LICENSE.txt", "file_count_records100": len(list((root / "records100").rglob("*.dat"))) + len(list((root / "records100").rglob("*.hea"))), "counts": counts, "metadata_sha256": {name: sha256_file(root / name) for name in ("ptbxl_database.csv", "scp_statements.csv", "LICENSE.txt", "SHA256SUMS.txt")}, "prepared_sha256": {name: sha256_file(prepared / name) for name in ("signals.npy", "labels.npy", "folds.npy", "ecg_ids.npy", "label_construction.csv", "normalization.json")}, "validation_status": "verified"}
    manifest["manifest_sha256"] = sha256_json(manifest)
    (prepared / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.verify_sha:
        expected = {}
        for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            digest, path = line.split(maxsplit=1); expected[path.lstrip("* ")] = digest
        for path in sorted((root / "records100").rglob("*")):
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                if expected.get(relative) != sha256_file(path):
                    raise ValueError(f"official checksum mismatch: {relative}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
