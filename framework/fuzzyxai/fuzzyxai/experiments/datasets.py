"""Deterministic controlled datasets for the multimodal validation contour.

The generators are intentionally local and license-clean. They validate protocol
behavior at scale, not external-domain generalization. External dataset runs can
replace them through the same ``BenchmarkDataset`` contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


Modality = Literal["tabular", "image", "text", "time_series"]


@dataclass(frozen=True)
class BenchmarkDataset:
    dataset_id: str
    modality: Modality
    values: np.ndarray | tuple[str, ...]
    labels: np.ndarray
    object_ids: np.ndarray
    feature_names: tuple[str, ...]
    critical_mask: np.ndarray
    rare_subgroup_mask: np.ndarray
    metadata: dict[str, object]

    @property
    def n_objects(self) -> int:
        return int(len(self.labels))


def _object_ids(prefix: str, n_objects: int) -> np.ndarray:
    return np.asarray([f"{prefix}_{index:06d}" for index in range(n_objects)])


def controlled_tabular(n_objects: int = 10_000, seed: int = 4201) -> BenchmarkDataset:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_objects, 16))
    rare = (values[:, 0] > 1.25) & (values[:, 1] < -0.8)
    logits = 1.4 * values[:, 0] - 1.1 * values[:, 1] + 0.8 * values[:, 2] + 0.45 * values[:, 3]
    logits += rare * (1.8 * values[:, 4] - 0.9)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.random(n_objects) < probabilities).astype(int)
    critical = rare & (labels == 1)
    return BenchmarkDataset(
        dataset_id="controlled_tabular_risk_v1",
        modality="tabular",
        values=values,
        labels=labels,
        object_ids=_object_ids("tab", n_objects),
        feature_names=tuple(f"feature_{index:02d}" for index in range(values.shape[1])),
        critical_mask=critical,
        rare_subgroup_mask=rare,
        metadata=_controlled_metadata(seed, "binary classification with a predefined rare interaction subgroup"),
    )


def controlled_images(n_objects: int = 10_000, seed: int = 4202, size: int = 16) -> BenchmarkDataset:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n_objects)
    images = rng.normal(0.0, 0.13, size=(n_objects, size, size))
    rare = rng.random(n_objects) < 0.08
    for index, label in enumerate(labels):
        if label == 0:
            images[index, 3:13, 6:10] += 0.85
        else:
            images[index, 6:10, 3:13] += 0.85
        if rare[index]:
            images[index] = np.rot90(images[index], 1)
            images[index, 1:4, 1:4] += 0.35
    images = np.clip(images, 0.0, 1.0).astype(np.float32)
    critical = rare.copy()
    return BenchmarkDataset(
        dataset_id="controlled_geometric_images_v1",
        modality="image",
        values=images,
        labels=labels.astype(int),
        object_ids=_object_ids("img", n_objects),
        feature_names=tuple(f"pixel_{row:02d}_{column:02d}" for row in range(size) for column in range(size)),
        critical_mask=critical,
        rare_subgroup_mask=rare,
        metadata=_controlled_metadata(seed, "binary geometric-image classification with a rotated rare subgroup"),
    )


def controlled_text(n_objects: int = 10_000, seed: int = 4203) -> BenchmarkDataset:
    rng = np.random.default_rng(seed)
    positive = ("fracture", "water", "pressure", "seepage", "unstable", "close")
    negative = ("dry", "stable", "distant", "sealed", "low", "normal")
    neutral = ("section", "sample", "survey", "layer", "measurement", "record", "zone", "depth")
    labels = rng.integers(0, 2, size=n_objects)
    rare = rng.random(n_objects) < 0.07
    documents: list[str] = []
    for label, is_rare in zip(labels, rare):
        class_words = positive if label else negative
        words = [str(rng.choice(class_words)) for _ in range(4)]
        words.extend(str(rng.choice(neutral)) for _ in range(8))
        if is_rare:
            words.extend(("legacy", "exception", str(rng.choice(negative if label else positive))))
        rng.shuffle(words)
        documents.append(" ".join(words))
    return BenchmarkDataset(
        dataset_id="controlled_geological_notes_v1",
        modality="text",
        values=tuple(documents),
        labels=labels.astype(int),
        object_ids=_object_ids("txt", n_objects),
        feature_names=tuple(sorted(set(positive + negative + neutral + ("legacy", "exception")))),
        critical_mask=rare.copy(),
        rare_subgroup_mask=rare,
        metadata=_controlled_metadata(seed, "binary text classification with contradictory rare notes"),
    )


def controlled_time_series(
    n_objects: int = 10_000,
    seed: int = 4204,
    window: int = 64,
) -> BenchmarkDataset:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n_objects)
    rare = rng.random(n_objects) < 0.08
    time = np.linspace(0.0, 2.0 * np.pi, window, endpoint=False)
    values = np.empty((n_objects, window), dtype=np.float32)
    for index, (label, is_rare) in enumerate(zip(labels, rare)):
        frequency = 1.0 if label == 0 else 2.0
        signal = np.sin(frequency * time + rng.uniform(-0.3, 0.3))
        signal += rng.normal(0.0, 0.12, size=window)
        if is_rare:
            signal[28:36] += 1.1 if label else -1.1
            signal = np.roll(signal, 5)
        values[index] = signal
    return BenchmarkDataset(
        dataset_id="controlled_sensor_windows_v1",
        modality="time_series",
        values=values,
        labels=labels.astype(int),
        object_ids=_object_ids("ts", n_objects),
        feature_names=tuple(f"time_{index:03d}" for index in range(window)),
        critical_mask=rare.copy(),
        rare_subgroup_mask=rare,
        metadata=_controlled_metadata(seed, "binary sensor-window classification with shifted rare events"),
    )


def _controlled_metadata(seed: int, task: str) -> dict[str, object]:
    return {
        "source_type": "controlled_synthetic",
        "source": "deterministic FuzzyXAI empirical-validation generator",
        "license": "CC0-1.0",
        "version": "1.0",
        "acquisition_date": "generated_at_runtime",
        "random_seed": seed,
        "task": task,
        "claim_scope": "protocol and controlled-behavior validation only; no external-domain generalization",
    }


def build_all_controlled(n_objects: int = 10_000) -> tuple[BenchmarkDataset, ...]:
    return (
        controlled_tabular(n_objects=n_objects),
        controlled_images(n_objects=n_objects),
        controlled_text(n_objects=n_objects),
        controlled_time_series(n_objects=n_objects),
    )


def snapshot_dataset(dataset: BenchmarkDataset, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(dataset.values, tuple):
        data_path = output_dir / f"{dataset.dataset_id}.jsonl"
        with data_path.open("w", encoding="utf-8") as handle:
            for object_id, text, label in zip(dataset.object_ids, dataset.values, dataset.labels):
                handle.write(json.dumps({"object_id": str(object_id), "text": text, "label": int(label)}, sort_keys=True) + "\n")
    else:
        data_path = output_dir / f"{dataset.dataset_id}.npz"
        np.savez_compressed(
            data_path,
            values=dataset.values,
            labels=dataset.labels,
            object_ids=dataset.object_ids,
            critical_mask=dataset.critical_mask,
            rare_subgroup_mask=dataset.rare_subgroup_mask,
        )
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    return {
        "dataset_id": dataset.dataset_id,
        "modality": dataset.modality,
        "n_objects": dataset.n_objects,
        "n_features": len(dataset.feature_names),
        "positive_rate": float(np.mean(dataset.labels)),
        "rare_subgroup_rate": float(np.mean(dataset.rare_subgroup_mask)),
        "critical_rate": float(np.mean(dataset.critical_mask)),
        "snapshot": str(data_path),
        "sha256": digest,
        **dataset.metadata,
    }
