"""Verify the official PAPILA v2 payload before any training is permitted.

The source files stay under ``FUZZYXAI_CH6_DATA_ROOT``.  This verifier writes
only derived manifests there, so raw medical images and clinical metadata can
never enter the repository by accident.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from chapter6_medical_validation.ophthalmology.src.artifact_io import sha256_file, sha256_json
from chapter6_medical_validation.ophthalmology.src.datasets import configured_data_root


def _papila_root(data_root: Path) -> Path:
    bases = sorted((data_root / "eyes" / "papila" / "raw").glob("PapilaDB-PAPILA-*"))
    if len(bases) != 1:
        raise FileNotFoundError("expected exactly one extracted PapilaDB-PAPILA-* directory")
    return bases[0]


def _clinical_rows(path: Path, eye: str) -> dict[str, dict[str, str]]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment-specific dependency
        raise RuntimeError("PAPILA verification requires pandas/openpyxl to read official XLSX metadata") from exc
    frame = pd.read_excel(path, header=None)
    records: dict[str, dict[str, str]] = {}
    for _, row in frame.iloc[3:].iterrows():
        raw_id = str(row.iloc[0]).strip()
        if not raw_id.startswith("#"):
            continue
        patient_id = f"RET{int(raw_id[1:]):03d}"
        diagnosis = row.iloc[3]
        if str(diagnosis).strip() not in {"0", "1", "2", "0.0", "1.0", "2.0"}:
            raise ValueError(f"unexpected diagnosis for {patient_id}{eye}: {diagnosis!r}")
        records[patient_id] = {"patient_id": patient_id, "eye": eye, "diagnosis": str(int(float(diagnosis)))}
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def verify(data_root: Path, output: Path | None = None) -> dict[str, Any]:
    root = _papila_root(data_root)
    output = output or (data_root / "eyes" / "papila" / "verified")
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = data_root / "eyes" / "papila" / "figshare_article_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    by_eye = {
        "OD": _clinical_rows(root / "ClinicalData" / "patient_data_od.xlsx", "OD"),
        "OS": _clinical_rows(root / "ClinicalData" / "patient_data_os.xlsx", "OS"),
    }
    labels = {f"{patient}{eye}": row for eye, table in by_eye.items() for patient, row in table.items()}
    images = sorted((root / "FundusImages").glob("*.jpg"))
    rows: list[dict[str, Any]] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    dimensions: Counter[str] = Counter(); invalid: list[dict[str, str]] = []
    for image in images:
        sample_id = image.stem
        eye = sample_id[-2:] if sample_id[-2:] in {"OD", "OS"} else ""
        patient_id = sample_id[:-2]
        label = labels.get(sample_id)
        try:
            with Image.open(image) as opened:
                opened.verify()
            with Image.open(image) as opened:
                size, mode = opened.size, opened.mode
            dimensions[f"{size[0]}x{size[1]}:{mode}"] += 1
        except Exception as exc:  # noqa: BLE001: manifest must disclose every bad image
            invalid.append({"sample_id": sample_id, "reason": f"{type(exc).__name__}: {exc}"})
            size = (0, 0); mode = "invalid"
        digest = sha256_file(image)
        hashes[digest].append(sample_id)
        contour_prefix = sample_id
        contours = root / "ExpertsSegmentations" / "Contours"
        expected = [contours / f"{contour_prefix}_{structure}_exp{expert}.txt" for structure in ("disc", "cup") for expert in (1, 2)]
        rows.append({
            "sample_id": sample_id, "patient_id": patient_id, "eye": eye,
            "diagnosis": label["diagnosis"] if label else "",
            "image_path": image.relative_to(data_root).as_posix(), "image_sha256": digest,
            "width": size[0], "height": size[1], "mode": mode,
            "expert_1_complete": str(all(path.is_file() for path in expected if "exp1" in path.name)).lower(),
            "expert_2_complete": str(all(path.is_file() for path in expected if "exp2" in path.name)).lower(),
        })
    contour_files = sorted((root / "ExpertsSegmentations" / "Contours").glob("*.txt"))
    patient_summary: list[dict[str, Any]] = []
    for patient_id in sorted({row["patient_id"] for row in rows} | set(by_eye["OD"]) | set(by_eye["OS"])):
        od, os = by_eye["OD"].get(patient_id), by_eye["OS"].get(patient_id)
        patient_summary.append({
            "patient_id": patient_id,
            "od_diagnosis": od["diagnosis"] if od else "",
            "os_diagnosis": os["diagnosis"] if os else "",
            "mixed_eye_diagnoses": str(bool(od and os and od["diagnosis"] != os["diagnosis"])).lower(),
            "suspect_associated": str(bool((od and od["diagnosis"] == "2") or (os and os["diagnosis"] == "2"))).lower(),
        })
    fields = ["sample_id", "patient_id", "eye", "diagnosis", "image_path", "image_sha256", "width", "height", "mode", "expert_1_complete", "expert_2_complete"]
    _write_csv(output / "papila_eye_labels.csv", rows, fields)
    _write_csv(output / "papila_patient_summary.csv", patient_summary, list(patient_summary[0]) if patient_summary else ["patient_id"])
    inventory = "\n".join(f"{row['image_sha256']}  {row['image_path']}" for row in rows) + "\n"
    (output / "papila_inventory_sha256.txt").write_text(inventory, encoding="utf-8")
    diagnoses = Counter(row["diagnosis"] for row in rows if row["diagnosis"])
    manifest = {
        "schema_version": "1.0", "status": "PASS" if not invalid and not [row for row in rows if not row["diagnosis"]] else "FAIL",
        "dataset": {"article_id": metadata.get("id"), "version": metadata.get("version"), "title": metadata.get("title"), "license": metadata.get("license", {}).get("name"), "license_url": metadata.get("license", {}).get("url")},
        "raw_root": root.relative_to(data_root).as_posix(), "patients": len(patient_summary), "eye_images": len(rows),
        "eyes": dict(Counter(row["eye"] for row in rows)), "diagnosis_counts": dict(sorted(diagnoses.items())),
        "mixed_eye_diagnosis_patients": sum(row["mixed_eye_diagnoses"] == "true" for row in patient_summary),
        "suspect_associated_patients": sum(row["suspect_associated"] == "true" for row in patient_summary),
        "images_missing_diagnosis": sum(not row["diagnosis"] for row in rows),
        "images_missing_expert_1_segmentation": sum(row["expert_1_complete"] != "true" for row in rows),
        "images_missing_expert_2_segmentation": sum(row["expert_2_complete"] != "true" for row in rows),
        "segmentation_contours": {"total": len(contour_files), "expert_1": sum("_exp1" in item.name for item in contour_files), "expert_2": sum("_exp2" in item.name for item in contour_files)},
        "duplicate_hash_groups": {digest: members for digest, members in hashes.items() if len(members) > 1},
        "invalid_images": invalid, "dimensions": dict(dimensions),
        "source_files": {"figshare_metadata_sha256": sha256_file(metadata_path), "od_clinical_sha256": sha256_file(root / "ClinicalData" / "patient_data_od.xlsx"), "os_clinical_sha256": sha256_file(root / "ClinicalData" / "patient_data_os.xlsx")},
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    (output / "papila_dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify official PAPILA files and write data-root-only manifests")
    parser.add_argument("--data-root")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = verify(configured_data_root(args.data_root), args.output)
    print(json.dumps({key: manifest[key] for key in ("status", "patients", "eye_images", "diagnosis_counts", "segmentation_contours")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
