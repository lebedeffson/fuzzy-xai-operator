#!/usr/bin/env python3
"""Fail closed when content-equivalent objects cross confirmatory splits."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.fft import dctn

from common import ROOT, STUDY, write


DATA_ROOT = ROOT / "data/confirmatory"
SPLITS = ("train", "development", "sealed_test")
TARGET_COLUMNS = {"target", "y", "label", "labels"}
MINHASH_SIZE = 64
MINHASH_BANDS = 16
TEXT_JACCARD_THRESHOLD = 0.90
IMAGE_PHASH_DISTANCE = 4


def main() -> None:
    leakage = json.loads((STUDY / "final_leakage_audit.json").read_text(encoding="utf-8"))
    if leakage.get("status") != "pass":
        raise SystemExit("BLOCKED: final leakage audit must pass before near-duplicate audit")

    reports = [
        _audit_tabular("bank_marketing", ignored={"object_id_hash"}),
        _audit_tabular("default_credit_clients", ignored={"object_id_hash", "ID"}),
        _audit_image("shoulder_implant_xray"),
        _audit_text("sms_spam"),
        _audit_timeseries("uci_har_smartphones"),
    ]
    violations = [item for report in reports for item in report["violations"]]
    payload = {
        "schema_version": "1.0",
        "status": "pass" if not violations else "blocked",
        "scope": "content_identity_and_near_duplicate_isolation_across_confirmatory_splits",
        "sealed_test_labels_loaded": False,
        "thresholds": {
            "image_phash_max_hamming_distance": IMAGE_PHASH_DISTANCE,
            "text_minhash_components": MINHASH_SIZE,
            "text_candidate_bands": MINHASH_BANDS,
            "text_exact_jaccard_threshold": TEXT_JACCARD_THRESHOLD,
        },
        "datasets": reports,
        "near_duplicate_violations": len(violations),
        "blockers": violations,
    }
    write(STUDY / "near_duplicate_audit.json", payload)
    if violations:
        raise SystemExit(f"BLOCKED: near-duplicate violations={len(violations)}; see near_duplicate_audit.json")
    print("PASS: final_near_duplicate_audit datasets=5 violations=0 labels_loaded=false")


def _audit_tabular(dataset_id: str, *, ignored: set[str]) -> dict[str, object]:
    records: list[tuple[str, str, str]] = []
    columns: list[str] | None = None
    for split in SPLITS:
        frame = pd.read_csv(DATA_ROOT / dataset_id / f"processed/{split}.csv")
        feature_columns = sorted(set(frame.columns) - ignored - TARGET_COLUMNS)
        columns = columns or feature_columns
        if columns != feature_columns:
            raise SystemExit(f"FAIL: inconsistent feature schema for {dataset_id}:{split}")
        for _, row in frame.iterrows():
            canonical = [_canonical_scalar(row[column]) for column in feature_columns]
            digest = _digest(json.dumps(canonical, ensure_ascii=True, separators=(",", ":")))
            records.append((split, str(row["object_id_hash"]), digest))
    violations = _cross_split_digest_violations(dataset_id, "exact_feature_hash", records)
    return {
        "dataset_id": dataset_id,
        "modality": "tabular",
        "method": "sha256_of_canonical_feature_row_excluding_identifier_and_target",
        "feature_count": len(columns or ()),
        "objects": len(records),
        "violations": violations,
    }


def _audit_image(dataset_id: str) -> dict[str, object]:
    rows: list[tuple[str, str, str, int]] = []
    for split in SPLITS:
        with np.load(DATA_ROOT / dataset_id / f"processed/{split}.npz") as payload:
            for image, object_id in zip(payload["x"], payload["object_id_hash"], strict=True):
                exact = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
                rows.append((split, str(object_id), exact, _phash(image)))
    exact_violations = _cross_split_digest_violations(
        dataset_id,
        "exact_pixel_hash",
        [(split, object_id, digest) for split, object_id, digest, _ in rows],
    )
    near_violations: list[dict[str, object]] = []
    for left_index, right_index in combinations(range(len(rows)), 2):
        left, right = rows[left_index], rows[right_index]
        if left[0] == right[0] or left[2] == right[2]:
            continue
        distance = (left[3] ^ right[3]).bit_count()
        if distance <= IMAGE_PHASH_DISTANCE:
            near_violations.append(
                _pair_violation(dataset_id, "perceptual_hash", left[0], left[1], right[0], right[1], distance)
            )
    return {
        "dataset_id": dataset_id,
        "modality": "image",
        "method": "exact_pixel_sha256_plus_64bit_dct_phash",
        "group_metadata": "unavailable_in_source_dataset",
        "objects": len(rows),
        "violations": exact_violations + near_violations,
    }


def _audit_text(dataset_id: str) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        frame = pd.read_csv(DATA_ROOT / dataset_id / f"processed/{split}.csv")
        for _, row in frame.iterrows():
            normalized = _normalize_text(str(row["text"]))
            shingles = _text_shingles(normalized)
            rows.append(
                {
                    "split": split,
                    "object_id": str(row["object_id_hash"]),
                    "digest": _digest(normalized),
                    "shingles": shingles,
                    "signature": _minhash(shingles),
                }
            )
    exact_violations = _cross_split_digest_violations(
        dataset_id,
        "normalized_text_hash",
        [(str(row["split"]), str(row["object_id"]), str(row["digest"])) for row in rows],
    )
    candidate_pairs: set[tuple[int, int]] = set()
    band_size = MINHASH_SIZE // MINHASH_BANDS
    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        signature = np.asarray(row["signature"], dtype=np.uint64)
        for band in range(MINHASH_BANDS):
            start = band * band_size
            buckets[(band, signature[start : start + band_size].tobytes())].append(index)
    for indices in buckets.values():
        for left, right in combinations(indices, 2):
            if rows[left]["split"] != rows[right]["split"]:
                candidate_pairs.add((left, right))
    near_violations: list[dict[str, object]] = []
    exact_pairs = {
        tuple(sorted((str(item["left_object_id"]), str(item["right_object_id"])))) for item in exact_violations
    }
    for left_index, right_index in sorted(candidate_pairs):
        left, right = rows[left_index], rows[right_index]
        identity_pair = tuple(sorted((str(left["object_id"]), str(right["object_id"]))))
        if identity_pair in exact_pairs:
            continue
        left_shingles = left["shingles"]
        right_shingles = right["shingles"]
        union = len(left_shingles | right_shingles)
        similarity = len(left_shingles & right_shingles) / union if union else 1.0
        if similarity >= TEXT_JACCARD_THRESHOLD:
            near_violations.append(
                _pair_violation(
                    dataset_id,
                    "minhash_candidate_exact_jaccard",
                    str(left["split"]),
                    str(left["object_id"]),
                    str(right["split"]),
                    str(right["object_id"]),
                    round(similarity, 6),
                )
            )
    return {
        "dataset_id": dataset_id,
        "modality": "text",
        "method": "normalized_sha256_plus_64_component_minhash_lsh_with_exact_jaccard_confirmation",
        "objects": len(rows),
        "candidate_pairs": len(candidate_pairs),
        "violations": exact_violations + near_violations,
    }


def _audit_timeseries(dataset_id: str) -> dict[str, object]:
    rows: list[tuple[str, str, str]] = []
    subjects_by_split: dict[str, set[int]] = {}
    for split in SPLITS:
        with np.load(DATA_ROOT / dataset_id / f"processed/{split}.npz") as payload:
            subjects_by_split[split] = {int(value) for value in payload["subject_id"]}
            for sequence, object_id in zip(payload["x"], payload["object_id_hash"], strict=True):
                digest = hashlib.sha256(np.ascontiguousarray(sequence).tobytes()).hexdigest()
                rows.append((split, str(object_id), digest))
    violations = _cross_split_digest_violations(dataset_id, "exact_sequence_hash", rows)
    for left, right in combinations(SPLITS, 2):
        overlap = subjects_by_split[left] & subjects_by_split[right]
        for subject in sorted(overlap):
            violations.append(
                {
                    "dataset_id": dataset_id,
                    "kind": "subject_overlap",
                    "left_split": left,
                    "right_split": right,
                    "subject_hash": _digest(f"{dataset_id}:subject:{subject}"),
                }
            )
    return {
        "dataset_id": dataset_id,
        "modality": "timeseries",
        "method": "subject_disjointness_plus_exact_float32_sequence_sha256",
        "objects": len(rows),
        "subjects_per_split": {key: len(value) for key, value in subjects_by_split.items()},
        "violations": violations,
    }


def _cross_split_digest_violations(
    dataset_id: str,
    kind: str,
    rows: list[tuple[str, str, str]],
) -> list[dict[str, object]]:
    by_digest: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split, object_id, digest in rows:
        by_digest[digest].append((split, object_id))
    violations: list[dict[str, object]] = []
    for digest, matches in by_digest.items():
        for left, right in combinations(matches, 2):
            if left[0] != right[0]:
                violations.append(_pair_violation(dataset_id, kind, left[0], left[1], right[0], right[1], digest))
    return violations


def _pair_violation(
    dataset_id: str,
    kind: str,
    left_split: str,
    left_object_id: str,
    right_split: str,
    right_object_id: str,
    measurement: object,
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "kind": kind,
        "left_split": left_split,
        "left_object_id": left_object_id,
        "right_split": right_split,
        "right_object_id": right_object_id,
        "measurement": measurement,
    }


def _canonical_scalar(value: object) -> object:
    if pd.isna(value):
        return {"missing": True}
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return format(float(value), ".17g")
    return str(value).strip()


def _phash(image: np.ndarray) -> int:
    array = np.asarray(image, dtype=np.float64)
    height, width = array.shape
    array = array[: height - (height % 32), : width - (width % 32)]
    pooled = array.reshape(32, array.shape[0] // 32, 32, array.shape[1] // 32).mean(axis=(1, 3))
    coefficients = dctn(pooled, type=2, norm="ortho")[:8, :8]
    threshold = np.median(coefficients.ravel()[1:])
    bits = coefficients.ravel() >= threshold
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[\w']+", value.casefold(), flags=re.UNICODE))


def _text_shingles(value: str) -> set[str]:
    tokens = value.split()
    if len(tokens) >= 3:
        return {" ".join(tokens[index : index + 3]) for index in range(len(tokens) - 2)}
    if len(value) >= 5:
        return {value[index : index + 5] for index in range(len(value) - 4)}
    return {value}


def _minhash(shingles: set[str]) -> np.ndarray:
    prime = np.uint64(2**61 - 1)
    indices = np.arange(1, MINHASH_SIZE + 1, dtype=np.uint64)
    multipliers = indices * np.uint64(2_654_435_761) + np.uint64(1)
    offsets = indices * np.uint64(805_459_861) + np.uint64(97)
    signature = np.full(MINHASH_SIZE, prime, dtype=np.uint64)
    for shingle in shingles:
        base = np.uint64(int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest(), "big") % int(prime))
        signature = np.minimum(signature, (multipliers * base + offsets) % prime)
    return signature


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
