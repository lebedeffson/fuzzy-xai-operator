from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import secrets
import subprocess
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

from .common import AMENDMENT, ARTIFACTS, DATA, git_commit, read_json, sha256_file, verify_protocol, write_json


SOURCES = {
    "uci_dry_bean": {
        "url": "https://archive.ics.uci.edu/static/public/602/data.csv",
        "filename": "data.csv",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode",
        "modality": "tabular",
    },
    "uci_news_aggregator": {
        "url": "https://archive.ics.uci.edu/static/public/359/news+aggregator.zip",
        "filename": "news_aggregator.zip",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode",
        "modality": "text",
    },
    "eurosat_rgb": {
        "url": "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1",
        "filename": "EuroSAT_RGB.zip",
        "license": "MIT",
        "license_url": "https://github.com/phelber/EuroSAT#license",
        "modality": "image",
    },
}


def _download(url: str, output: Path) -> None:
    if output.is_file() and output.stat().st_size > 0:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "FuzzyXAI-independent-confirmatory/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
    temporary.replace(output)


def _split_for_group(group_id: str) -> str:
    bucket = int(hashlib.sha256(f"independent-v1:{group_id}".encode()).hexdigest()[:12], 16) / float(16**12)
    if bucket < 0.60:
        return "train"
    if bucket < 0.80:
        return "formative_development"
    if bucket < 0.90:
        return "sealed_calibration_check"
    return "sealed_confirmatory_test"


def _dry_bean(path: Path) -> Iterable[tuple[str, str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "Class" not in reader.fieldnames:
            raise RuntimeError("Dry Bean schema is missing Class")
        for index, row in enumerate(reader):
            object_id = f"dry-bean:{index:05d}"
            yield object_id, str(row["Class"]), object_id


def _news(path: Path) -> Iterable[tuple[str, str, str]]:
    with zipfile.ZipFile(path) as archive:
        candidates = [name for name in archive.namelist() if name.lower().endswith((".csv", ".txt")) and "readme" not in name.lower()]
        if not candidates:
            raise RuntimeError("News Aggregator archive contains no corpus file")
        with archive.open(max(candidates, key=lambda name: archive.getinfo(name).file_size)) as binary:
            for index, raw in enumerate(binary):
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                fields = line.split("\t")
                if len(fields) < 8:
                    continue
                object_id = f"news:{fields[0] or index}"
                label = fields[4]
                story_group = fields[5] or object_id
                yield object_id, label, f"story:{story_group}"


def _eurosat(path: Path) -> Iterable[tuple[str, str, str]]:
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                continue
            parts = Path(name).parts
            if len(parts) < 2:
                continue
            payload = archive.read(name)
            content_hash = hashlib.sha256(payload).hexdigest()
            yield f"eurosat:{content_hash[:20]}", parts[-2], f"content:{content_hash}"


def _encrypt_labels(dataset_dir: Path, labels: dict[str, str]) -> tuple[str, str]:
    private = dataset_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    plaintext = private / "labels.json"
    encrypted = private / "confirmatory_label_vault.enc"
    key = private / ".vault_passphrase"
    if not key.exists():
        key.write_text(secrets.token_hex(32), encoding="ascii")
        os.chmod(key, 0o600)
    plaintext.write_text(json.dumps(labels, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    subprocess.run(
        ["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-salt", "-in", str(plaintext), "-out", str(encrypted), "-pass", f"file:{key}"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    plaintext.unlink()
    return sha256_file(encrypted), sha256_file(key)


def _prepare_dataset(dataset_id: str, metadata: dict[str, str], *, download: bool) -> dict[str, object]:
    dataset_dir = DATA / dataset_id
    raw = dataset_dir / "raw" / metadata["filename"]
    if download:
        _download(metadata["url"], raw)
    if not raw.is_file():
        return {"dataset_id": dataset_id, "status": "pending_download", "sealed": False, **metadata}
    parser = {"uci_dry_bean": _dry_bean, "uci_news_aggregator": _news, "eurosat_rgb": _eurosat}[dataset_id]
    rows = list(parser(raw))
    if not rows:
        raise RuntimeError(f"{dataset_id} produced no identities")
    groups = {}
    labels = {}
    object_splits = {}
    class_counts: dict[str, Counter[str]] = {}
    for object_id, label, group_id in rows:
        split = groups.setdefault(group_id, _split_for_group(group_id))
        object_splits[object_id] = split
        class_counts.setdefault(split, Counter())[label] += 1
        if split in {"sealed_calibration_check", "sealed_confirmatory_test"}:
            labels[object_id] = label
    group_splits: dict[str, set[str]] = {}
    for object_id, _, group_id in rows:
        group_splits.setdefault(group_id, set()).add(object_splits[object_id])
    overlap = sum(len(values) > 1 for values in group_splits.values())
    vault_hash, key_hash = _encrypt_labels(dataset_dir, labels)
    identity_lists = {
        split: sorted(object_id for object_id, value in object_splits.items() if value == split)
        for split in ("train", "formative_development", "sealed_calibration_check", "sealed_confirmatory_test")
    }
    public = {
        "dataset_id": dataset_id,
        "modality": metadata["modality"],
        "source": metadata["url"],
        "license": metadata["license"],
        "license_url": metadata["license_url"],
        "raw_sha256": sha256_file(raw),
        "n_objects": len(rows),
        "n_groups": len(groups),
        "split_counts": {name: len(values) for name, values in identity_lists.items()},
        "class_counts": {split: dict(sorted(counts.items())) for split, counts in class_counts.items()},
        "split_identity_sha256": {split: hashlib.sha256("\n".join(values).encode()).hexdigest() for split, values in identity_lists.items()},
        "group_overlap_count": overlap,
        "content_duplicate_groups_kept_within_split": True,
        "sealed_label_vault_sha256": vault_hash,
        "vault_key_sha256_recorded_locally_only": key_hash,
        "test_labels_in_public_manifest": False,
        "normalization_fit_scope": "train_only",
        "sealed": overlap == 0,
        "status": "sealed_data_identity_and_label_vault" if overlap == 0 else "blocked_group_overlap",
    }
    local_manifest = dataset_dir / "manifests" / "dataset_manifest.json"
    write_json(local_manifest, public)
    write_json(dataset_dir / "manifests" / "split_identities.json", identity_lists)
    return public


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    protocol_hashes = verify_protocol()
    amendment = read_json(AMENDMENT)
    effective_ids = [item["dataset_id"] for item in amendment["effective_independent_datasets"]]
    if effective_ids != list(SOURCES):
        raise RuntimeError("dataset implementation differs from registered amendment")
    datasets = [_prepare_dataset(dataset_id, SOURCES[dataset_id], download=args.download) for dataset_id in effective_ids]
    all_sealed = all(item.get("sealed", False) for item in datasets)
    manifest = {
        "schema_version": "1.0",
        "protocol_hashes": protocol_hashes,
        "implementation_commit": git_commit(),
        "datasets": datasets,
        "all_datasets_sealed": all_sealed,
        "confirmatory_opening_allowed": False,
        "reason": "Data sealing does not open labels or freeze models, heads, costs, baselines, or policies.",
    }
    write_json(ARTIFACTS / "data" / "dataset_manifest.json", manifest)
    write_json(
        ARTIFACTS / "data" / "leakage_audit.json",
        {
            "status": "pass" if all_sealed else "blocked_pending_download_or_overlap",
            "group_overlap_count": sum(int(item.get("group_overlap_count", 0)) for item in datasets),
            "test_labels_in_public_artifacts": False,
            "vault_encryption": "AES-256-CBC PBKDF2 via OpenSSL",
            "vault_keys_versioned": False,
            "confirmatory_test_opened": False,
        },
    )
    print(f"independent-data status={'SEALED' if all_sealed else 'PENDING'} datasets={len(datasets)} confirmatory_opening=false")


if __name__ == "__main__":
    main()
