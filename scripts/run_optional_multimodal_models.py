#!/usr/bin/env python3
"""Measure CNN/ONNX and sequence-model channels for E1."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from fuzzyxai.experiments.datasets import controlled_images, controlled_text, controlled_time_series
from fuzzyxai.experiments.metrics import binary_classification_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "release_evidence/full_empirical_validation/optional_runtime/multimodal_neural_runtime.json"


def encode_text(documents: tuple[str, ...], vocabulary: tuple[str, ...], max_length: int = 18) -> np.ndarray:
    indices = {token: index + 1 for index, token in enumerate(vocabulary)}
    encoded = np.zeros((len(documents), max_length), dtype=np.int64)
    for row_index, document in enumerate(documents):
        for column_index, token in enumerate(document.split()[:max_length]):
            encoded[row_index, column_index] = indices.get(token, 0)
    return encoded


def split_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=seed)
    return np.sort(train), np.sort(test)


def train_classifier(
    model: Any,
    train_values: Any,
    train_labels: Any,
    test_values: Any,
    *,
    epochs: int,
    batch_size: int,
) -> tuple[np.ndarray, float, float]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = torch.nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(train_values, train_labels), batch_size=batch_size, shuffle=True, num_workers=0)
    start = perf_counter()
    model.train()
    for _ in range(epochs):
        for batch, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch), labels)
            loss.backward()
            optimizer.step()
    fit_seconds = perf_counter() - start
    model.eval()
    start = perf_counter()
    with torch.no_grad():
        probabilities = torch.softmax(model(test_values), dim=1)[:, 1].cpu().numpy()
    predict_seconds = perf_counter() - start
    return probabilities, fit_seconds, predict_seconds


def run(n_objects: int, output: Path, seed: int = 42) -> dict[str, object]:
    import onnxruntime as ort
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(1)

    class TinyCNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(1, 6, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(6, 10, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((2, 2)),
            )
            self.output = torch.nn.Linear(40, 2)

        def forward(self, values: Any) -> Any:
            return self.output(self.features(values).flatten(1))

    class TextGRU(torch.nn.Module):
        def __init__(self, vocabulary_size: int) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(vocabulary_size + 1, 12, padding_idx=0)
            self.gru = torch.nn.GRU(12, 12, batch_first=True)
            self.output = torch.nn.Linear(12, 2)

        def forward(self, values: Any) -> Any:
            _, state = self.gru(self.embedding(values))
            return self.output(state[-1])

    class SeriesGRU(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = torch.nn.GRU(1, 12, batch_first=True)
            self.output = torch.nn.Linear(12, 2)

        def forward(self, values: Any) -> Any:
            _, state = self.gru(values)
            return self.output(state[-1])

    rows: list[dict[str, object]] = []

    images = controlled_images(n_objects=n_objects)
    train, test = split_indices(images.labels, seed)
    image_train = torch.tensor(np.asarray(images.values)[train, None, :, :], dtype=torch.float32)
    image_test = torch.tensor(np.asarray(images.values)[test, None, :, :], dtype=torch.float32)
    cnn = TinyCNN()
    probabilities, fit_seconds, predict_seconds = train_classifier(
        cnn,
        image_train,
        torch.tensor(images.labels[train], dtype=torch.long),
        image_test,
        epochs=3,
        batch_size=128,
    )
    onnx_path = output.parent / "controlled_image_cnn.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        cnn,
        image_test[:1],
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {"input": image_test[:64].numpy()})[0]
    onnx_probabilities = np.exp(onnx_logits - onnx_logits.max(axis=1, keepdims=True))
    onnx_probabilities /= onnx_probabilities.sum(axis=1, keepdims=True)
    parity_error = float(np.max(np.abs(onnx_probabilities[:, 1] - probabilities[:64])))
    rows.append(
        {
            "dataset_id": images.dataset_id,
            "modality": "image",
            "model": "pytorch_tiny_cnn",
            "status": "measured",
            "n_objects": n_objects,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "metrics": binary_classification_metrics(images.labels[test], probabilities, subgroup_mask=images.rare_subgroup_mask[test]),
            "channels": ["gradient_map", "integrated_gradients_compatible", "perturbation_stability"],
            "onnx_export": str(onnx_path),
            "onnx_probability_max_abs_error": parity_error,
            "onnx_parity_pass": parity_error <= 1e-5,
        }
    )

    text = controlled_text(n_objects=n_objects)
    text_values = encode_text(text.values, text.feature_names)
    train, test = split_indices(text.labels, seed)
    text_model = TextGRU(len(text.feature_names))
    probabilities, fit_seconds, predict_seconds = train_classifier(
        text_model,
        torch.tensor(text_values[train], dtype=torch.long),
        torch.tensor(text.labels[train], dtype=torch.long),
        torch.tensor(text_values[test], dtype=torch.long),
        epochs=3,
        batch_size=128,
    )
    rows.append(
        {
            "dataset_id": text.dataset_id,
            "modality": "text",
            "model": "pytorch_gru",
            "status": "measured",
            "n_objects": n_objects,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "metrics": binary_classification_metrics(text.labels[test], probabilities, subgroup_mask=text.rare_subgroup_mask[test]),
            "channels": ["token_masking", "sequence_sensitivity", "paraphrase_stability_protocol"],
        }
    )

    series = controlled_time_series(n_objects=n_objects)
    train, test = split_indices(series.labels, seed)
    series_model = SeriesGRU()
    probabilities, fit_seconds, predict_seconds = train_classifier(
        series_model,
        torch.tensor(np.asarray(series.values)[train, :, None], dtype=torch.float32),
        torch.tensor(series.labels[train], dtype=torch.long),
        torch.tensor(np.asarray(series.values)[test, :, None], dtype=torch.float32),
        epochs=3,
        batch_size=128,
    )
    rows.append(
        {
            "dataset_id": series.dataset_id,
            "modality": "time_series",
            "model": "pytorch_gru",
            "status": "measured",
            "n_objects": n_objects,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "metrics": binary_classification_metrics(series.labels[test], probabilities, subgroup_mask=series.rare_subgroup_mask[test]),
            "channels": ["window_ablation", "shift_stability", "state_history"],
        }
    )
    payload = {
        "schema_version": "1.0",
        "result_origin": "measured_on_controlled_datasets",
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "onnxruntime": ort.__version__},
        "runs": rows,
        "checks": {
            "cnn_measured": any(row["model"] == "pytorch_tiny_cnn" and row["status"] == "measured" for row in rows),
            "onnx_parity": bool(rows[0]["onnx_parity_pass"]),
            "text_sequence_measured": any(row["modality"] == "text" and row["status"] == "measured" for row in rows),
            "time_series_sequence_measured": any(row["modality"] == "time_series" and row["status"] == "measured" for row in rows),
        },
        "limitations": ["controlled modality generators", "maps are not causal"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS: optional_multimodal_neural_runtime")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.objects, arguments.output)
