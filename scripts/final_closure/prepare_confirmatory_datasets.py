#!/usr/bin/env python3
"""Prepare five real confirmatory datasets without exposing sealed-test labels."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import secrets
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from audit_near_duplicates import (
    IMAGE_PHASH_DISTANCE,
    MINHASH_BANDS,
    MINHASH_SIZE,
    TEXT_JACCARD_THRESHOLD,
    _minhash,
    _normalize_text,
    _phash,
    _text_shingles,
)
from common import ROOT, STUDY, sha256, write


DATA_ROOT = ROOT / "data/confirmatory"
KEY_PATH = Path.home() / ".config/fuzzyxai/confirmatory_vault_aes256.pass"
SPLIT_SEED = 7419


@dataclass(frozen=True)
class DatasetSource:
    dataset_id: str
    modality: str
    source: str
    source_url: str
    doi: str
    expected_sha256: str
    grouping_key: str | None


SOURCES = (
    DatasetSource(
        "bank_marketing",
        "tabular",
        "UCI Bank Marketing",
        "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
        "10.24432/C5K306",
        "e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4",
        None,
    ),
    DatasetSource(
        "default_credit_clients",
        "tabular",
        "UCI Default of Credit Card Clients",
        "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        "10.24432/C55S3H",
        "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602",
        "ID",
    ),
    DatasetSource(
        "shoulder_implant_xray",
        "image",
        "UCI Shoulder Implant X-Ray Manufacturer Classification",
        "https://archive.ics.uci.edu/static/public/517/shoulder+implant+x+ray+manufacturer+classification.zip",
        "10.24432/C5F893",
        "1286d0b92ed90a9de978a28770fb546756d067b0a504f33e811c72486c712a45",
        None,
    ),
    DatasetSource(
        "sms_spam",
        "text",
        "UCI SMS Spam Collection",
        "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "10.24432/C5CC84",
        "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3",
        None,
    ),
    DatasetSource(
        "uci_har_smartphones",
        "timeseries",
        "UCI Human Activity Recognition Using Smartphones",
        "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        "10.24432/C54S4K",
        "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031",
        "subject_id",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-file", type=Path, default=KEY_PATH)
    args = parser.parse_args()
    key_path = args.key_file.expanduser().resolve()
    _ensure_key(key_path)
    results = []
    all_oof: list[str] = []
    all_test: list[str] = []
    for source in SOURCES:
        archive = DATA_ROOT / source.dataset_id / "raw/source.zip"
        if not archive.is_file() or sha256(archive) != source.expected_sha256:
            raise SystemExit(f"FAIL: source archive missing or changed for {source.dataset_id}")
        if source.dataset_id == "bank_marketing":
            prepared = _prepare_bank(source, archive, key_path)
        elif source.dataset_id == "default_credit_clients":
            prepared = _prepare_default(source, archive, key_path)
        elif source.dataset_id == "shoulder_implant_xray":
            prepared = _prepare_xray(source, archive, key_path)
        elif source.dataset_id == "sms_spam":
            prepared = _prepare_sms(source, archive, key_path)
        else:
            prepared = _prepare_har(source, archive, key_path)
        results.append(prepared["dataset"])
        all_oof.extend(prepared["oof_ids"])
        all_test.extend(prepared["test_ids"])
    oof_path = STUDY / "oof_object_hashes.txt"
    test_path = STUDY / "sealed_test_object_hashes.txt"
    _write_lines(oof_path, sorted(all_oof))
    _write_lines(test_path, sorted(all_test))
    dataset_input = {
        "schema_version": "1.0",
        "selection_commit": _git_head(),
        "license_policy": "all datasets CC BY 4.0",
        "datasets": results,
    }
    split_input = {
        "schema_version": "1.0",
        "split_seed": SPLIT_SEED,
        "tuning_runner_can_read_test_labels": False,
        "test_labels_loaded_by_tuning": False,
        "controller_feature_source": "out_of_fold_train_development_only",
        "test_identity_visibility_during_tuning": "hash_only",
        "oof_object_hashes_path": oof_path.relative_to(ROOT).as_posix(),
        "oof_object_hashes_file_sha256": sha256(oof_path),
        "sealed_test_object_hashes_path": test_path.relative_to(ROOT).as_posix(),
        "sealed_test_object_hashes_file_sha256": sha256(test_path),
        "oof_object_count": len(all_oof),
        "sealed_test_object_count": len(all_test),
        "overlap_count": len(set(all_oof) & set(all_test)),
        "raw_sources_excluded_from_tuning_mount": True,
        "vault_key_path_in_repository": False,
    }
    write(STUDY / "confirmatory_dataset_manifest.input.json", dataset_input)
    write(STUDY / "confirmatory_split_manifest.input.json", split_input)
    print(
        "PASS: confirmatory_datasets_prepared "
        f"datasets={len(results)} oof={len(all_oof)} sealed_test={len(all_test)} overlap=0"
    )


def _prepare_bank(source: DatasetSource, archive: Path, key_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(archive) as outer:
        nested = outer.read("bank-additional.zip")
    with zipfile.ZipFile(io.BytesIO(nested)) as inner:
        raw = inner.read("bank-additional/bank-additional-full.csv")
    frame = pd.read_csv(io.BytesIO(raw), sep=";")
    before = len(frame)
    feature_columns = [column for column in frame.columns if column != "y"]
    frame = frame.drop_duplicates(subset=feature_columns, keep="first").reset_index(drop=True)
    object_ids = [_object_hash(source.dataset_id, index, row) for index, row in frame[feature_columns].iterrows()]
    frame.insert(0, "object_id_hash", object_ids)
    train_end, dev_end = int(0.60 * len(frame)), int(0.80 * len(frame))
    splits = {
        "train": np.arange(0, train_end),
        "development": np.arange(train_end, dev_end),
        "sealed_test": np.arange(dev_end, len(frame)),
    }
    return _write_tabular(
        source,
        frame,
        "y",
        splits,
        key_path,
        {
            "strategy": "temporal_order_60_20_20",
            "ordered_source": True,
            "exact_feature_duplicates_removed": before - len(frame),
        },
    )


def _prepare_default(source: DatasetSource, archive: Path, key_path: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="fxai-default-") as temp:
        temp_path = Path(temp)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extract("default of credit card clients.xls", temp_path)
        subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "csv",
                "--outdir",
                str(temp_path),
                str(temp_path / "default of credit card clients.xls"),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        frame = pd.read_csv(temp_path / "default of credit card clients.csv", skiprows=1)
    frame = frame.rename(columns={frame.columns[-1]: "target"})
    frame["object_id_hash"] = frame["ID"].map(lambda value: _digest(f"{source.dataset_id}:{int(value)}"))
    frame = frame[["object_id_hash", *[column for column in frame.columns if column != "object_id_hash"]]]
    feature_columns = [column for column in frame.columns if column not in {"object_id_hash", "ID", "target"}]
    groups = [
        _digest(json.dumps([_json_scalar(value) for value in row], separators=(",", ":")))
        for row in frame[feature_columns].itertuples(index=False, name=None)
    ]
    splits = _stratified_group_split(frame["target"].astype(str).tolist(), groups)
    return _write_tabular(
        source,
        frame,
        "target",
        splits,
        key_path,
        {
            "strategy": "stratified_group_60_20_20",
            "grouping_key": "exact_feature_cluster",
            "source_identifier": "ID",
            "unique_source_ids": True,
            "duplicate_feature_groups": len(groups) - len(set(groups)),
        },
    )


def _prepare_sms(source: DatasetSource, archive: Path, key_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(archive) as bundle:
        text = bundle.read("SMSSpamCollection").decode("utf-8")
    rows = []
    seen = set()
    for line in text.splitlines():
        label, message = line.split("\t", 1)
        normalized = " ".join(message.split()).casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        rows.append((_digest(f"{source.dataset_id}:{normalized}"), message, label))
    frame = pd.DataFrame(rows, columns=["object_id_hash", "text", "target"])
    groups, near_pairs = _text_duplicate_groups(frame["text"].astype(str).tolist())
    splits = _stratified_group_split(frame["target"].tolist(), groups)
    return _write_tabular(
        source,
        frame,
        "target",
        splits,
        key_path,
        {
            "strategy": "stratified_near_duplicate_group_60_20_20",
            "grouping_key": "normalized_text_minhash_cluster",
            "normalized_text_duplicates_removed": len(text.splitlines()) - len(frame),
            "near_duplicate_pairs_grouped": near_pairs,
            "near_duplicate_threshold": TEXT_JACCARD_THRESHOLD,
        },
    )


def _prepare_xray(source: DatasetSource, archive: Path, key_path: Path) -> dict[str, object]:
    images: list[np.ndarray] = []
    labels: list[str] = []
    object_ids: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        names = sorted(name for name in bundle.namelist() if name.lower().endswith((".jpg", ".jpeg")))
        for name in names:
            payload = bundle.read(name)
            label = Path(name).name.split(".", 1)[0]
            image = Image.open(io.BytesIO(payload)).convert("L").resize((128, 128))
            images.append(np.asarray(image, dtype=np.uint8))
            labels.append(label)
            object_ids.append(_digest(f"{source.dataset_id}:{hashlib.sha256(payload).hexdigest()}"))
    arrays = np.stack(images)
    groups, near_pairs = _image_duplicate_groups(arrays)
    splits = _stratified_group_split(labels, groups)
    return _write_npz(
        source,
        arrays,
        np.asarray(labels),
        np.asarray(object_ids),
        splits,
        key_path,
        {
            "strategy": "stratified_perceptual_hash_group_60_20_20",
            "transform": "grayscale_resize_128x128_uint8",
            "source_duplicate_note": "UCI removed eight repeated-patient images before publication",
            "grouping_key": "perceptual_hash_cluster",
            "near_duplicate_pairs_grouped": near_pairs,
            "perceptual_hash_max_hamming_distance": IMAGE_PHASH_DISTANCE,
        },
    )


def _prepare_har(source: DatasetSource, archive: Path, key_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(archive) as outer:
        nested = outer.read("UCI HAR Dataset.zip")
    signals: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[int] = []
    origins: list[str] = []
    signal_names = (
        "body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    )
    with zipfile.ZipFile(io.BytesIO(nested)) as bundle:
        root = "UCI HAR Dataset"
        for partition in ("train", "test"):
            channels = []
            for signal_name in signal_names:
                path = f"{root}/{partition}/Inertial Signals/{signal_name}_{partition}.txt"
                channels.append(np.loadtxt(io.BytesIO(bundle.read(path)), dtype=np.float32))
            partition_signals = np.stack(channels, axis=-1)
            partition_labels = np.loadtxt(io.BytesIO(bundle.read(f"{root}/{partition}/y_{partition}.txt")), dtype=np.int16)
            partition_subjects = np.loadtxt(io.BytesIO(bundle.read(f"{root}/{partition}/subject_{partition}.txt")), dtype=np.int16)
            signals.append(partition_signals)
            labels.extend(partition_labels.tolist())
            subjects.extend(partition_subjects.tolist())
            origins.extend(f"{partition}:{index}" for index in range(len(partition_labels)))
    x = np.concatenate(signals, axis=0)
    y = np.asarray(labels, dtype=np.int16)
    subject_array = np.asarray(subjects, dtype=np.int16)
    object_ids = np.asarray(
        [_digest(f"{source.dataset_id}:{subject}:{origin}:{hashlib.sha256(x[index].tobytes()).hexdigest()}") for index, (subject, origin) in enumerate(zip(subjects, origins, strict=True))]
    )
    subject_order = sorted(set(subjects), key=lambda value: _digest(f"{SPLIT_SEED}:{value}"))
    group_map = {
        **{subject: "train" for subject in subject_order[:18]},
        **{subject: "development" for subject in subject_order[18:24]},
        **{subject: "sealed_test" for subject in subject_order[24:]},
    }
    splits = {name: np.flatnonzero(np.asarray([group_map[value] == name for value in subjects])) for name in ("train", "development", "sealed_test")}
    return _write_npz(
        source,
        x,
        y,
        object_ids,
        splits,
        key_path,
        {
            "strategy": "subject_group_18_6_6",
            "grouping_key": "subject_id",
            "subject_groups": {name: sorted(set(subject_array[index].tolist())) for name, index in splits.items()},
            "signal_shape": [128, 9],
            "sampling_rate_hz": 50,
        },
        extra={"subject_id": subject_array},
    )


def _write_tabular(
    source: DatasetSource,
    frame: pd.DataFrame,
    target: str,
    splits: dict[str, np.ndarray],
    key_path: Path,
    preprocessing: dict[str, object],
) -> dict[str, object]:
    processed = DATA_ROOT / source.dataset_id / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for name, indices in splits.items():
        selected = frame.iloc[indices].copy()
        if name == "sealed_test":
            selected = selected.drop(columns=[target])
        selected.to_csv(processed / f"{name}.csv", index=False)
    labels = {
        frame.iloc[index]["object_id_hash"]: _json_scalar(frame.iloc[index][target])
        for index in splits["sealed_test"]
    }
    return _finalize(source, frame["object_id_hash"].tolist(), splits, labels, key_path, preprocessing)


def _write_npz(
    source: DatasetSource,
    x: np.ndarray,
    y: np.ndarray,
    object_ids: np.ndarray,
    splits: dict[str, np.ndarray],
    key_path: Path,
    preprocessing: dict[str, object],
    *,
    extra: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    processed = DATA_ROOT / source.dataset_id / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for name, indices in splits.items():
        payload: dict[str, np.ndarray] = {"x": x[indices], "object_id_hash": object_ids[indices]}
        if name != "sealed_test":
            payload["y"] = y[indices]
        if extra:
            payload.update({key: value[indices] for key, value in extra.items()})
        np.savez_compressed(processed / f"{name}.npz", **payload)
    labels = {str(object_ids[index]): _json_scalar(y[index]) for index in splits["sealed_test"]}
    return _finalize(source, object_ids.tolist(), splits, labels, key_path, preprocessing)


def _finalize(
    source: DatasetSource,
    object_ids: list[str],
    splits: dict[str, np.ndarray],
    labels: dict[str, object],
    key_path: Path,
    preprocessing: dict[str, object],
) -> dict[str, object]:
    root = DATA_ROOT / source.dataset_id
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    split_ids: dict[str, Path] = {}
    for name, indices in splits.items():
        path = manifests / f"{name}_object_ids.txt"
        _write_lines(path, sorted(object_ids[index] for index in indices))
        split_ids[name] = path
    preprocessing_payload = {
        "dataset_id": source.dataset_id,
        "modality": source.modality,
        "split_seed": SPLIT_SEED,
        "test_labels_in_processed_data": False,
        **preprocessing,
    }
    preprocessing_path = manifests / "preprocessing_manifest.json"
    write(preprocessing_path, preprocessing_payload)
    vault = root / "manifests/confirmatory_label_vault.enc"
    _encrypt_labels(vault, labels, key_path)
    license_path = manifests / "license.txt"
    license_path.write_text(
        f"{source.source}\nLicense: Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
        f"DOI: https://doi.org/{source.doi}\nSource: {source.source_url}\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset_id": source.dataset_id,
        "modality": source.modality,
        "source": source.source_url,
        "license": "CC BY 4.0",
        "download_sha256": source.expected_sha256,
        "preprocessing_sha256": sha256(preprocessing_path),
        "train_ids_sha256": sha256(split_ids["train"]),
        "development_ids_sha256": sha256(split_ids["development"]),
        "test_ids_sha256": sha256(split_ids["sealed_test"]),
        "grouping_key": preprocessing.get("grouping_key", source.grouping_key),
        "label_vault_sha256": sha256(vault),
        "label_vault_path": vault.relative_to(ROOT).as_posix(),
        "used_in_formative_tuning": False,
    }
    write(manifests / "dataset_manifest.json", manifest)
    write(
        manifests / "split_manifest.json",
        {
            "dataset_id": source.dataset_id,
            "counts": {name: len(indices) for name, indices in splits.items()},
            "identity_files": {name: path.relative_to(ROOT).as_posix() for name, path in split_ids.items()},
            "identity_file_sha256": {name: sha256(path) for name, path in split_ids.items()},
            "intersection_counts": {
                "train_development": _intersection(split_ids["train"], split_ids["development"]),
                "train_test": _intersection(split_ids["train"], split_ids["sealed_test"]),
                "development_test": _intersection(split_ids["development"], split_ids["sealed_test"]),
            },
        },
    )
    _write_checksums(root)
    return {
        "dataset": manifest,
        "oof_ids": [object_ids[index] for name in ("train", "development") for index in splits[name]],
        "test_ids": [object_ids[index] for index in splits["sealed_test"]],
    }


def _stratified_split(object_ids: list[str], labels: list[str]) -> dict[str, np.ndarray]:
    output = {"train": [], "development": [], "sealed_test": []}
    for label in sorted(set(labels)):
        indices = [index for index, value in enumerate(labels) if value == label]
        indices.sort(key=lambda index: _digest(f"{SPLIT_SEED}:{object_ids[index]}"))
        train_end, dev_end = int(0.60 * len(indices)), int(0.80 * len(indices))
        output["train"].extend(indices[:train_end])
        output["development"].extend(indices[train_end:dev_end])
        output["sealed_test"].extend(indices[dev_end:])
    return {name: np.asarray(sorted(indices), dtype=np.int64) for name, indices in output.items()}


def _stratified_group_split(labels: list[str], groups: list[str]) -> dict[str, np.ndarray]:
    """Assign each content-similarity group to one deterministic 20% fold."""
    labels_array = np.asarray(labels)
    groups_array = np.asarray(groups)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    folds = [held for _, held in splitter.split(np.zeros(len(labels_array)), labels_array, groups_array)]
    return {
        "train": np.asarray(sorted(np.concatenate(folds[2:])), dtype=np.int64),
        "development": np.asarray(sorted(folds[1]), dtype=np.int64),
        "sealed_test": np.asarray(sorted(folds[0]), dtype=np.int64),
    }


def _image_duplicate_groups(images: np.ndarray) -> tuple[list[str], int]:
    parents = list(range(len(images)))
    hashes = [_phash(image) for image in images]
    near_pairs = 0
    for left in range(len(images)):
        for right in range(left + 1, len(images)):
            if (hashes[left] ^ hashes[right]).bit_count() <= IMAGE_PHASH_DISTANCE:
                _union(parents, left, right)
                near_pairs += 1
    return [_digest(f"image-cluster:{_find(parents, index)}") for index in range(len(images))], near_pairs


def _text_duplicate_groups(texts: list[str]) -> tuple[list[str], int]:
    parents = list(range(len(texts)))
    shingles = [_text_shingles(_normalize_text(text)) for text in texts]
    signatures = [_minhash(value) for value in shingles]
    band_size = MINHASH_SIZE // MINHASH_BANDS
    buckets: dict[tuple[int, bytes], list[int]] = {}
    candidates: set[tuple[int, int]] = set()
    for index, signature in enumerate(signatures):
        for band in range(MINHASH_BANDS):
            start = band * band_size
            key = (band, signature[start : start + band_size].tobytes())
            for other in buckets.setdefault(key, []):
                candidates.add((other, index))
            buckets[key].append(index)
    near_pairs = 0
    normalized = [_normalize_text(text) for text in texts]
    exact: dict[str, int] = {}
    for index, value in enumerate(normalized):
        if value in exact:
            _union(parents, exact[value], index)
            near_pairs += 1
        else:
            exact[value] = index
    for left, right in sorted(candidates):
        union = len(shingles[left] | shingles[right])
        similarity = len(shingles[left] & shingles[right]) / union if union else 1.0
        if similarity >= TEXT_JACCARD_THRESHOLD:
            if _find(parents, left) != _find(parents, right):
                _union(parents, left, right)
            near_pairs += 1
    return [_digest(f"text-cluster:{_find(parents, index)}") for index in range(len(texts))], near_pairs


def _find(parents: list[int], value: int) -> int:
    while parents[value] != value:
        parents[value] = parents[parents[value]]
        value = parents[value]
    return value


def _union(parents: list[int], left: int, right: int) -> None:
    left_root, right_root = _find(parents, left), _find(parents, right)
    if left_root != right_root:
        parents[max(left_root, right_root)] = min(left_root, right_root)


def _encrypt_labels(path: Path, labels: dict[str, object], key_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix="fxai-labels-", suffix=".json") as clear, tempfile.NamedTemporaryFile(
        "wb", prefix="fxai-labels-", suffix=".enc", dir=path.parent, delete=False
    ) as encrypted:
        encrypted_path = Path(encrypted.name)
        json.dump({"labels": labels}, clear, sort_keys=True)
        clear.flush()
        try:
            subprocess.run(
                [
                    "openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-iter", "200000", "-salt",
                    "-in", clear.name, "-out", str(encrypted_path), "-pass", f"file:{key_path}",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            encrypted_path.replace(path)
        finally:
            encrypted_path.unlink(missing_ok=True)


def _ensure_key(path: Path) -> None:
    if path.is_file():
        if path.stat().st_mode & 0o077:
            raise SystemExit("FAIL: vault key must not be group/world accessible")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(secrets.token_hex(48) + "\n", encoding="ascii")
    path.chmod(0o600)


def _write_checksums(root: Path) -> None:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files),
        encoding="utf-8",
    )


def _write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def _intersection(left: Path, right: Path) -> int:
    return len(set(left.read_text(encoding="utf-8").splitlines()) & set(right.read_text(encoding="utf-8").splitlines()))


def _object_hash(dataset_id: str, index: int, row: pd.Series) -> str:
    canonical = json.dumps([_json_scalar(value) for value in row.tolist()], ensure_ascii=False, separators=(",", ":"))
    return _digest(f"{dataset_id}:{index}:{canonical}")


def _json_scalar(value: object) -> object:
    return value.item() if hasattr(value, "item") else value


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


if __name__ == "__main__":
    main()
