#!/usr/bin/env python3
"""Build public aliases and feature schemas without reading sealed labels."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import ROOT, STUDY, sha256, write


DATA_ROOT = ROOT / "data/confirmatory"
ALIASES = {
    "bank_marketing": "bank_marketing",
    "default_credit": "default_credit_clients",
    "shoulder_xray": "shoulder_implant_xray",
    "sms_spam": "sms_spam",
    "uci_har": "uci_har_smartphones",
}
MODALITIES = {
    "bank_marketing": "tabular",
    "default_credit_clients": "tabular",
    "shoulder_implant_xray": "image",
    "sms_spam": "text",
    "uci_har_smartphones": "timeseries",
}
TARGET_NAMES = {
    "bank_marketing": "y",
    "default_credit_clients": "target",
    "sms_spam": "target",
}


def main() -> None:
    leakage = json.loads((STUDY / "final_leakage_audit.json").read_text(encoding="utf-8"))
    if leakage.get("status") != "pass":
        raise SystemExit("BLOCKED: leakage audit must pass before metadata build")
    write(
        STUDY / "dataset_alias_manifest.json",
        {
            "schema_version": "1.0",
            "policy": "spec_aliases_map_to_stable_historical_dataset_ids_without_renaming_evidence",
            "aliases": ALIASES,
        },
    )
    reports = []
    for dataset_id, modality in MODALITIES.items():
        schema = _schema(dataset_id, modality)
        output = STUDY / f"dataset_manifests/{dataset_id}/feature_schema_manifest.json"
        write(output, schema)
        root = DATA_ROOT / dataset_id
        required = {
            "dataset_manifest": root / "manifests/dataset_manifest.json",
            "license": root / "manifests/license.txt",
            "preprocessing_manifest": root / "manifests/preprocessing_manifest.json",
            "split_manifest": root / "manifests/split_manifest.json",
            "feature_schema_manifest": output,
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            raise SystemExit(f"FAIL: missing metadata for {dataset_id}: {missing}")
        reports.append(
            {
                "dataset_id": dataset_id,
                "modality": modality,
                "files": {
                    name: {
                        "path": path.relative_to(ROOT).as_posix(),
                        "sha256": sha256(path),
                    }
                    for name, path in required.items()
                },
                "model_manifest_status": "pending_real_oof_model_fit",
                "explainer_manifest_status": "pending_real_oof_explanations",
            }
        )
    write(
        STUDY / "confirmatory_data_metadata_manifest.json",
        {
            "schema_version": "1.0",
            "status": "pass",
            "sealed_test_labels_loaded": False,
            "datasets": reports,
        },
    )
    print("PASS: final_data_metadata datasets=5 aliases=5 labels_loaded=false")


def _schema(dataset_id: str, modality: str) -> dict[str, object]:
    root = DATA_ROOT / dataset_id / "processed"
    if modality in {"tabular", "text"}:
        frame = pd.read_csv(root / "train.csv", nrows=200)
        target = TARGET_NAMES[dataset_id]
        fields = []
        for name, dtype in frame.dtypes.items():
            if name in {"object_id_hash", target}:
                continue
            fields.append(
                {
                    "name": name,
                    "dtype": str(dtype),
                    "missing_allowed": bool(frame[name].isna().any()),
                    "role": "unstructured_text" if modality == "text" else "predictor",
                }
            )
        return {
            "schema_version": "1.0",
            "dataset_id": dataset_id,
            "modality": modality,
            "identity_field": "object_id_hash",
            "target_field_visible_in_sealed_test": False,
            "fields": fields,
        }
    with np.load(root / "train.npz") as payload:
        x = payload["x"]
        extra = sorted(set(payload.files) - {"x", "y", "object_id_hash"})
    value = {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "modality": modality,
        "identity_field": "object_id_hash",
        "target_field_visible_in_sealed_test": False,
        "tensor": {"name": "x", "sample_shape": list(x.shape[1:]), "dtype": str(x.dtype)},
        "group_fields": extra,
    }
    if dataset_id == "shoulder_implant_xray":
        value["image_contract"] = {"color_space": "grayscale", "resolution": [128, 128], "normalization": "uint8_0_255"}
    if dataset_id == "uci_har_smartphones":
        value["timeseries_contract"] = {
            "window_samples": 128,
            "channels": 9,
            "sampling_rate_hz": 50,
            "group_field": "subject_id",
        }
    return value


if __name__ == "__main__":
    main()
