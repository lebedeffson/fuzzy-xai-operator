"""Optional PyTorch native-modality benchmarks for the final heavy CI."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from .multiclass import NativeDataset, _ece, _hash, _stratified_cap, load_native_dataset


def run_neural_benchmark(
    modality: str,
    output: Path,
    cache: Path,
    *,
    seeds: Sequence[int] = (4201, 4202, 4203, 4204, 4205),
    epochs: int = 5,
    train_cap: int = 30_000,
) -> dict[str, object]:
    if modality not in {"image", "text", "timeseries"}:
        raise ValueError("neural benchmark supports image, text and timeseries")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("install the torch extra to run native neural benchmarks") from error

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split

    torch.set_num_threads(1)
    dataset = load_native_dataset(modality, cache)
    values = _neural_values(dataset)
    indices = np.arange(len(dataset.labels))
    runs = []
    exported_onnx: dict[str, object] | None = None
    frozen_checkpoint: dict[str, object] | None = None
    for seed in seeds:
        train_val, test = train_test_split(indices, test_size=0.2, random_state=seed, stratify=dataset.labels)
        train, validation = train_test_split(
            train_val,
            test_size=0.25,
            random_state=seed + 100,
            stratify=dataset.labels[train_val],
        )
        train = _stratified_cap(train, dataset.labels, train_cap, seed)
        validation = _stratified_cap(validation, dataset.labels, 5_000, seed + 1)
        test = _stratified_cap(test, dataset.labels, 5_000, seed + 2)
        for model_id in _model_ids(modality):
            torch.manual_seed(seed)
            model = _build_model(modality, model_id, values, int(len(np.unique(dataset.labels))))
            started = perf_counter()
            history = _train_model(
                model,
                values,
                dataset.labels,
                train,
                validation,
                epochs=epochs,
                seed=seed,
            )
            probabilities = _predict(model, values, test)
            predictions = probabilities.argmax(axis=1)
            labels = dataset.labels[test]
            run = {
                "seed": seed,
                "model_id": model_id,
                "family": _family(model_id),
                "n_train": len(train),
                "n_validation": len(validation),
                "n_test": len(test),
                "epochs": epochs,
                "fit_seconds": perf_counter() - started,
                "accuracy": float(accuracy_score(labels, predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
                "macro_f1": float(f1_score(labels, predictions, average="macro")),
                "micro_f1": float(f1_score(labels, predictions, average="micro")),
                "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
                "ece": _ece(probabilities, labels),
                "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
                "history": history,
                "split_hashes": {
                    "train": _hash(train),
                    "validation": _hash(validation),
                    "test": _hash(test),
                },
                "test_used_for_selection": False,
                "status": "measured",
            }
            runs.append(run)
            if seed == seeds[0] and model_id == _model_ids(modality)[0]:
                checkpoint_path = output.parent / f"{modality}_frozen_model.pt"
                torch.save(model.state_dict(), checkpoint_path)
                frozen_checkpoint = {
                    "path": checkpoint_path.name,
                    "model_id": model_id,
                    "seed": seed,
                    "train_sha256": _hash(train),
                    "validation_sha256": _hash(validation),
                    "test_sha256": _hash(test),
                    "state_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                }
            if modality == "image" and model_id == "compact_cnn" and seed == seeds[0]:
                exported_onnx = _export_and_verify_onnx(model, values[test[:8]], output.parent / "fashion_compact_cnn.onnx")
    payload = {
        "schema_version": "2.0",
        "dataset_id": dataset.dataset_id,
        "modality": modality,
        "native_class_count": int(len(np.unique(dataset.labels))),
        "seeds": list(seeds),
        "models": runs,
        "onnx": exported_onnx,
        "frozen_explainer_checkpoint": frozen_checkpoint,
        "status": "PASS",
        "limitations": [
            "compact research architectures are benchmark probes, not state-of-the-art models",
            "training caps and epoch counts are reported explicitly",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _neural_values(dataset: NativeDataset) -> np.ndarray:
    if dataset.modality == "text":
        documents = dataset.values
        if not isinstance(documents, tuple):
            raise TypeError("text dataset must contain documents")
        return np.asarray([_token_ids(document) for document in documents], dtype=np.int64)
    values = np.asarray(dataset.values, dtype=np.float32)
    if dataset.modality == "image":
        return values[:, None, :, :]
    return values[:, None, :]


def _token_ids(document: str, *, length: int = 128, vocabulary: int = 10_000) -> list[int]:
    tokens = re.findall(r"[a-zA-Z]{2,}", document.lower())[:length]
    result = [1 + int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % (vocabulary - 1) for token in tokens]
    return result + [0] * (length - len(result))


def _model_ids(modality: str) -> tuple[str, ...]:
    return {
        "image": ("compact_cnn", "deeper_cnn"),
        "text": ("compact_gru",),
        "timeseries": ("one_dimensional_cnn", "timeseries_gru", "temporal_convolutional_network"),
    }[modality]


def _family(model_id: str) -> str:
    if "gru" in model_id:
        return "sequence"
    if "convolutional" in model_id:
        return "TCN"
    return "CNN"


def _build_model(modality: str, model_id: str, values: np.ndarray, classes: int) -> object:
    import torch

    class ImageCNN(torch.nn.Module):
        def __init__(self, deeper: bool) -> None:
            super().__init__()
            layers: list[torch.nn.Module] = [
                torch.nn.Conv2d(1, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(16, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
            ]
            if deeper:
                layers.extend((torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU()))
            self.features = torch.nn.Sequential(*layers)
            self.pool = torch.nn.AdaptiveAvgPool2d((2, 2))
            self.head = torch.nn.Linear((64 if deeper else 32) * 4, classes)

        def forward(self, inputs: object) -> object:
            return self.head(self.pool(self.features(inputs)).flatten(1))

    class SequenceGRU(torch.nn.Module):
        def __init__(self, text: bool) -> None:
            super().__init__()
            self.text = text
            self.embedding = torch.nn.Embedding(10_000, 32, padding_idx=0) if text else None
            self.gru = torch.nn.GRU(32 if text else 1, 32, batch_first=True)
            self.head = torch.nn.Linear(32, classes)

        def forward(self, inputs: object) -> object:
            sequence = self.embedding(inputs) if self.embedding is not None else inputs.transpose(1, 2)
            _, hidden = self.gru(sequence)
            return self.head(hidden[-1])

    class TimeConv(torch.nn.Module):
        def __init__(self, tcn: bool) -> None:
            super().__init__()
            dilation = 2 if tcn else 1
            self.features = torch.nn.Sequential(
                torch.nn.Conv1d(1, 16, 5, padding=2),
                torch.nn.ReLU(),
                torch.nn.Conv1d(16, 32, 3, padding=dilation, dilation=dilation),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1),
            )
            self.head = torch.nn.Linear(32, classes)

        def forward(self, inputs: object) -> object:
            return self.head(self.features(inputs).squeeze(-1))

    if modality == "image":
        return ImageCNN(model_id == "deeper_cnn")
    if model_id in {"compact_gru", "timeseries_gru"}:
        return SequenceGRU(modality == "text")
    return TimeConv(model_id == "temporal_convolutional_network")


def _train_model(
    model: object,
    values: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    *,
    epochs: int,
    seed: int,
) -> list[dict[str, float | int]]:
    import torch

    generator = torch.Generator().manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(values[train]), torch.from_numpy(labels[train]).long())
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch_values, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_values), batch_labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_probabilities = _predict(model, values, validation)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_accuracy": float(np.mean(validation_probabilities.argmax(axis=1) == labels[validation])),
            }
        )
    return history


def _predict(model: object, values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            batch = torch.from_numpy(values[indices[start : start + 512]])
            chunks.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(chunks)


def _export_and_verify_onnx(model: object, sample: np.ndarray, output: Path) -> dict[str, object]:
    try:
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as error:
        return {"status": "not_installed", "reason": repr(error)}
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    tensor = torch.from_numpy(sample)
    torch.onnx.export(
        model,
        tensor,
        output,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    onnx.checker.check_model(onnx.load(output))
    native = model(tensor).detach().numpy()
    runtime = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    exported = runtime.run(None, {"input": sample})[0]
    difference = float(np.max(np.abs(native - exported)))
    return {
        "status": "verified" if difference <= 1e-4 else "failed",
        "path": output.name,
        "max_absolute_difference": difference,
    }
