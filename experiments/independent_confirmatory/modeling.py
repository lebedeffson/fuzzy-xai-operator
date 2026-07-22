from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import ARTIFACTS, DATA, PRIVATE, git_commit, read_json, sha256_file, stable_unit_interval, verify_protocol, write_json


DATASET_IDS = ("uci_dry_bean", "uci_news_aggregator", "eurosat_rgb")


def _split_lookup(dataset_id: str) -> dict[str, str]:
    values = read_json(DATA / dataset_id / "manifests" / "split_identities.json")
    return {object_id: split for split, identities in values.items() for object_id in identities}


def _dry_rows() -> Iterator[tuple[str, np.ndarray, str]]:
    path = DATA / "uci_dry_bean" / "raw" / "data.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        names = tuple(name for name in (reader.fieldnames or ()) if name != "Class")
        for index, row in enumerate(reader):
            yield f"dry-bean:{index:05d}", np.asarray([float(row[name]) for name in names]), str(row["Class"])


def _news_rows() -> Iterator[tuple[str, str, str]]:
    path = DATA / "uci_news_aggregator" / "raw" / "news_aggregator.zip"
    with zipfile.ZipFile(path) as archive:
        candidates = [name for name in archive.namelist() if name.lower().endswith((".csv", ".txt")) and "readme" not in name.lower()]
        name = max(candidates, key=lambda item: archive.getinfo(item).file_size)
        with archive.open(name) as binary:
            for index, raw in enumerate(binary):
                fields = raw.decode("utf-8", errors="replace").rstrip("\r\n").split("\t")
                if len(fields) < 8:
                    continue
                yield f"news:{fields[0] or index}", f"{fields[1]} {fields[3]}", fields[4]


def _image_features(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        rgb = np.asarray(image.convert("RGB").resize((8, 8)), dtype=np.float32) / 255.0
    flat = rgb.reshape(-1)
    statistics = np.concatenate((rgb.mean((0, 1)), rgb.std((0, 1)), np.quantile(rgb, (0.25, 0.5, 0.75), axis=(0, 1)).reshape(-1)))
    return np.concatenate((flat, statistics)).astype(np.float32)


def _eurosat_rows() -> Iterator[tuple[str, np.ndarray, str]]:
    path = DATA / "eurosat_rgb" / "raw" / "EuroSAT_RGB.zip"
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                continue
            payload = archive.read(name)
            identity = hashlib.sha256(payload).hexdigest()
            yield f"eurosat:{identity[:20]}", _image_features(payload), Path(name).parts[-2]


def _load_all(dataset_id: str) -> tuple[dict[str, list[object]], dict[str, list[str]], dict[str, list[str]]]:
    lookup = _split_lookup(dataset_id)
    values: dict[str, list[object]] = {split: [] for split in set(lookup.values())}
    labels: dict[str, list[str]] = {split: [] for split in set(lookup.values())}
    ids: dict[str, list[str]] = {split: [] for split in set(lookup.values())}
    iterator = {"uci_dry_bean": _dry_rows, "uci_news_aggregator": _news_rows, "eurosat_rgb": _eurosat_rows}[dataset_id]()
    for object_id, value, label in iterator:
        split = lookup[object_id]
        ids[split].append(object_id)
        values[split].append(value)
        if split in {"train", "formative_development"}:
            labels[split].append(label)
    return values, labels, ids


def _fit(dataset_id: str, values: dict[str, list[object]], labels: dict[str, list[str]]) -> object:
    if dataset_id == "uci_dry_bean":
        model = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=31, learning_rate=0.08, random_state=7301)
        model.fit(np.asarray(values["train"], dtype=float), np.asarray(labels["train"]))
        return model
    if dataset_id == "uci_news_aggregator":
        model = make_pipeline(
            TfidfVectorizer(max_features=30_000, min_df=3, ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32),
            SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=30, tol=1e-4, random_state=7301, n_jobs=1),
        )
        model.fit(values["train"], np.asarray(labels["train"]))
        return model
    model = make_pipeline(
        StandardScaler(),
        SGDClassifier(loss="log_loss", alpha=2e-5, max_iter=250, tol=1e-4, random_state=7301, n_jobs=1),
    )
    model.fit(np.asarray(values["train"], dtype=float), np.asarray(labels["train"]))
    return model


def _predict(model: object, values: list[object], dataset_id: str) -> tuple[np.ndarray, np.ndarray]:
    matrix: object = values if dataset_id == "uci_news_aggregator" else np.asarray(values, dtype=float)
    probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float32)  # type: ignore[attr-defined]
    classes = np.asarray(model.classes_) if hasattr(model, "classes_") else np.asarray(model[-1].classes_)  # type: ignore[index]
    predictions = classes[np.argmax(probabilities, axis=1)]
    return predictions.astype(str), probabilities


def observable_rows(object_ids: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, np.ndarray]:
    ordered = np.sort(probabilities, axis=1)
    confidence = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-9, 1.0)), axis=1) / np.log(probabilities.shape[1])
    route_draw = np.asarray([stable_unit_interval(str(value), salt="independent-route-v1") for value in object_ids])
    repairable = (route_draw < 0.04) & (route_draw >= 0.005)
    irreparable = route_draw < 0.005
    route_fault = repairable | irreparable
    perturbation = np.asarray([stable_unit_interval(str(value), salt="independent-explanation-v1") for value in object_ids])
    explanation_risk = np.clip(0.65 * entropy + 0.35 * perturbation, 0.0, 1.0)
    shift_draw = np.asarray([stable_unit_interval(str(value), salt="independent-shift-v1") for value in object_ids])
    shift_risk = np.clip(0.70 * (1.0 - margin) + 0.30 * shift_draw, 0.0, 1.0)
    return {
        "object_ids": object_ids.astype(str),
        "predictions": predictions.astype(str),
        "probabilities": probabilities,
        "predictive_risk": (1.0 - confidence).astype(np.float32),
        "entropy": entropy.astype(np.float32),
        "margin_risk": (1.0 - margin).astype(np.float32),
        "route_risk": route_fault.astype(np.float32),
        "repairable_fault": repairable,
        "irreparable_fault": irreparable,
        "explanation_risk": explanation_risk.astype(np.float32),
        "shift_risk": shift_risk.astype(np.float32),
    }


def main() -> None:
    verify_protocol()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    summaries = []
    for dataset_id in DATASET_IDS:
        values, labels, ids = _load_all(dataset_id)
        model = _fit(dataset_id, values, labels)
        model_path = PRIVATE / f"{dataset_id}.joblib"
        joblib.dump(model, model_path)
        split_summaries = {}
        for split in ("formative_development", "sealed_calibration_check", "sealed_confirmatory_test"):
            prediction, probabilities = _predict(model, values[split], dataset_id)
            rows = observable_rows(np.asarray(ids[split]), prediction, probabilities)
            if split == "formative_development":
                rows["labels"] = np.asarray(labels[split])
            output = PRIVATE / f"{dataset_id}-{split}.npz"
            np.savez_compressed(output, **rows)
            split_summaries[split] = {"objects": len(prediction), "artifact_sha256": sha256_file(output), "labels_present": split == "formative_development"}
        summaries.append(
            {
                "dataset_id": dataset_id,
                "model_family": type(model[-1] if hasattr(model, "__getitem__") else model).__name__,
                "train_objects": len(ids["train"]),
                "development_objects": len(ids["formative_development"]),
                "model_sha256": sha256_file(model_path),
                "splits": split_summaries,
                "normalization_and_vocabulary_fit_scope": "train_only",
                "sealed_labels_loaded": False,
            }
        )
    write_json(
        ARTIFACTS / "models" / "model_manifest.json",
        {
            "schema_version": "1.0",
            "phase": "prelock_train_and_observable_prediction",
            "implementation_commit": git_commit(),
            "datasets": summaries,
            "test_labels_loaded": False,
            "route_faults": "deterministic controlled injection independent of labels; 3.5% repairable and 0.5% irreparable",
        },
    )
    print(f"PASS independent-models datasets={len(summaries)} sealed_labels_loaded=false")


if __name__ == "__main__":
    main()
