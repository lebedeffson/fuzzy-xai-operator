"""Real benchmark acquisition and modality-specific measurements.

The heavy CI invokes one modality per job. Raw data remain in the runner cache;
only hashes, cards and measured summaries become release evidence.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import struct
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LoadedBenchmark:
    dataset_id: str
    modality: str
    values: np.ndarray | tuple[str, ...]
    labels: np.ndarray
    source: str
    license: str
    version: str
    raw_sha256: str
    processed_sha256: str
    preprocessing: tuple[str, ...]
    limitations: tuple[str, ...]

    @property
    def n_objects(self) -> int:
        return len(self.labels)

    def card(self) -> dict[str, object]:
        n_features = self.values.shape[1] if isinstance(self.values, np.ndarray) and self.values.ndim > 1 else 1
        return {
            "dataset_id": self.dataset_id,
            "modality": self.modality,
            "task": "binary_classification_derived_from_original_labels",
            "source": self.source,
            "license": self.license,
            "version": self.version,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "raw_sha256": self.raw_sha256,
            "processed_sha256": self.processed_sha256,
            "n_objects": self.n_objects,
            "n_features": int(n_features),
            "target": "predeclared binary grouping",
            "preprocessing": list(self.preprocessing),
            "known_limitations": list(self.limitations),
        }


def load_real_benchmark(modality: str, cache: Path) -> LoadedBenchmark:
    cache.mkdir(parents=True, exist_ok=True)
    if modality == "tabular":
        return _load_covtype(cache)
    if modality == "image":
        return _load_fashion_mnist(cache)
    if modality == "text":
        return _load_20newsgroups(cache)
    if modality == "timeseries":
        return _load_electric_devices(cache)
    raise ValueError(f"unknown modality: {modality}")


def run_real_benchmark(modality: str, output: Path, cache: Path, *, seed: int = 4201) -> dict[str, object]:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    dataset = load_real_benchmark(modality, cache)
    indices = np.arange(dataset.n_objects)
    train, test = train_test_split(indices, test_size=0.2, stratify=dataset.labels, random_state=seed)
    train_cap = train[: min(20_000, len(train))]
    test_cap = test[: min(5_000, len(test))]
    features = _classical_features(dataset, fit_indices=train_cap)
    models: list[tuple[str, str, object]] = [
        ("logistic", "linear", make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))),
        ("random_forest", "tree", RandomForestClassifier(n_estimators=60, max_depth=12, n_jobs=1, random_state=seed)),
    ]
    if modality == "tabular":
        try:
            from xgboost import XGBClassifier

            models.append(("xgboost", "boosting", XGBClassifier(n_estimators=60, max_depth=5, learning_rate=0.08, n_jobs=1, random_state=seed)))
        except ImportError as error:
            raise RuntimeError("the tabular heavy job requires xgboost") from error
    else:
        models.append(("hist_gradient_boosting", "boosting", HistGradientBoostingClassifier(max_iter=80, random_state=seed)))
    runs: list[dict[str, object]] = []
    fitted: dict[str, object] = {}
    for model_id, family, model in models:
        start = perf_counter()
        model.fit(features[train_cap], dataset.labels[train_cap])
        fit_seconds = perf_counter() - start
        start = perf_counter()
        predictions = np.asarray(model.predict(features[test_cap]), dtype=int)
        predict_seconds = perf_counter() - start
        fitted[model_id] = model
        runs.append(
            {
                "model_id": model_id,
                "family": family,
                "library": "scikit-learn",
                "n_train": len(train_cap),
                "n_test": len(test_cap),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "accuracy": float(accuracy_score(dataset.labels[test_cap], predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(dataset.labels[test_cap], predictions)),
                "f1": float(f1_score(dataset.labels[test_cap], predictions, zero_division=0)),
                "status": "measured",
            }
        )
    explainers = _classical_explainers(dataset, features, train_cap, test_cap, fitted["logistic"], seed)
    explainers.extend(
        (
            {"method": "linear_coefficients", "status": "measured", "implementation": "sklearn.LogisticRegression"},
            {"method": "native_tree_path", "status": "measured", "implementation": "sklearn.RandomForestClassifier"},
        )
    )
    neural = _run_neural_channels(dataset, train, test, seed)
    payload = {
        "schema_version": "1.0",
        "dataset": dataset.card(),
        "models": runs + neural["models"],
        "explainers": explainers + neural["explainers"],
        "controlled_counterpart": f"controlled_{modality}",
        "evidence_origin": "measured_real_benchmark",
        "claim_scope": "framework empirical benchmark; no domain deployment claim",
        "medical_disclaimer": "The result is not a clinical validation and is not intended for autonomous medical decisions.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _load_covtype(cache: Path) -> LoadedBenchmark:
    from sklearn.datasets import fetch_covtype

    bunch = fetch_covtype(data_home=cache, as_frame=False, download_if_missing=True)
    values = np.asarray(bunch.data, dtype=np.float32)
    original = np.asarray(bunch.target, dtype=int)
    labels = (original <= 2).astype(int)
    raw_hash = _array_sha(values, original)
    return LoadedBenchmark(
        "uci_covertype",
        "tabular",
        values,
        labels,
        "https://archive.ics.uci.edu/dataset/31/covertype",
        "CC BY 4.0",
        "UCI-31/sklearn-cache",
        raw_hash,
        _array_sha(values, labels),
        ("float32 conversion", "binary target: original cover types 1-2 versus 3-7"),
        ("benchmark target is a preregistered binary derivative of the seven-class task",),
    )


def _load_fashion_mnist(cache: Path) -> LoadedBenchmark:
    base = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
    names = (
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    )
    paths = [_download(f"{base}/{name}", cache / name) for name in names]
    train_images = _read_idx_images(paths[0])
    train_labels = _read_idx_labels(paths[1])
    test_images = _read_idx_images(paths[2])
    test_labels = _read_idx_labels(paths[3])
    values = np.concatenate((train_images, test_images)).astype(np.float32) / 255.0
    original = np.concatenate((train_labels, test_labels))
    labels = np.isin(original, (5, 7, 9)).astype(int)
    return LoadedBenchmark(
        "fashion_mnist",
        "image",
        values,
        labels,
        "https://github.com/zalandoresearch/fashion-mnist",
        "MIT",
        "official-master-idx",
        _files_sha(paths),
        _array_sha(values, labels),
        ("scale pixels to [0,1]", "binary target: footwear classes 5,7,9 versus other apparel"),
        ("fashion benchmark; no medical-image generalization",),
    )


def _load_20newsgroups(cache: Path) -> LoadedBenchmark:
    from sklearn.datasets import fetch_20newsgroups

    bunch = fetch_20newsgroups(
        data_home=cache,
        subset="all",
        remove=("headers", "footers", "quotes"),
        shuffle=False,
        download_if_missing=True,
    )
    documents = tuple(str(item) for item in bunch.data)
    original = np.asarray(bunch.target, dtype=int)
    labels = (original < len(bunch.target_names) // 2).astype(int)
    encoded = "\n\0\n".join(documents).encode("utf-8", errors="replace")
    return LoadedBenchmark(
        "20newsgroups",
        "text",
        documents,
        labels,
        "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html",
        "source-specific post rights; scikit-learn loader is BSD-3-Clause",
        "sklearn-cache",
        hashlib.sha256(encoded + original.tobytes()).hexdigest(),
        hashlib.sha256(encoded + labels.tobytes()).hexdigest(),
        ("remove headers, footers and quotes", "binary target: first ten versus last ten categories"),
        ("historical posts can contain personal or offensive language", "raw text is not redistributed"),
    )


def _load_electric_devices(cache: Path) -> LoadedBenchmark:
    base = "https://zenodo.org/records/11190880/files"
    train_path = _download(f"{base}/ElectricDevices_TRAIN.ts?download=1", cache / "ElectricDevices_TRAIN.ts")
    test_path = _download(f"{base}/ElectricDevices_TEST.ts?download=1", cache / "ElectricDevices_TEST.ts")
    train_values, train_labels = _read_ts(train_path)
    test_values, test_labels = _read_ts(test_path)
    values = np.concatenate((train_values, test_values)).astype(np.float32)
    original = np.concatenate((train_labels, test_labels))
    unique = sorted(set(float(item) for item in original))
    labels = np.isin(original, unique[: max(1, len(unique) // 2)]).astype(int)
    return LoadedBenchmark(
        "ucr_electric_devices",
        "timeseries",
        values,
        labels,
        "https://zenodo.org/records/11190880",
        "CC BY 4.0",
        "Zenodo-v1",
        _files_sha((train_path, test_path)),
        _array_sha(values, labels),
        ("merge official train and test before preregistered re-split", "binary grouping of original seven classes"),
        ("household electricity benchmark; no operational energy claim",),
    )


def _classical_features(dataset: LoadedBenchmark, *, fit_indices: np.ndarray) -> np.ndarray:
    if dataset.modality == "text":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=256, min_df=2)
        vectorizer.fit([dataset.values[int(index)] for index in fit_indices])
        matrix = vectorizer.transform(dataset.values)
        return np.asarray(matrix.toarray(), dtype=np.float32)
    values = np.asarray(dataset.values, dtype=np.float32)
    if dataset.modality == "image":
        n = len(values)
        return values.reshape(n, 4, 7, 4, 7).mean(axis=(2, 4)).reshape(n, 16)
    if dataset.modality == "timeseries":
        spectrum = np.abs(np.fft.rfft(values, axis=1))[:, 1:17]
        summary = np.column_stack((values.mean(axis=1), values.std(axis=1), values.min(axis=1), values.max(axis=1)))
        return np.column_stack((spectrum, summary)).astype(np.float32)
    return values


def _classical_explainers(
    dataset: LoadedBenchmark,
    features: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    model: object,
    seed: int,
) -> list[dict[str, object]]:
    if dataset.modality != "tabular":
        method = "token_masking" if dataset.modality == "text" else "window_masking" if dataset.modality == "timeseries" else "occlusion"
        rows = [{"method": method, "status": "measured", "implementation": "fuzzyxai.q1_validation", "n_explained": min(20, len(test))}]
        try:
            import shap

            named_steps = getattr(model, "named_steps")
            scaler = named_steps["standardscaler"]
            estimator = named_steps["logisticregression"]
            background = scaler.transform(features[train[: min(200, len(train))]])
            sample = scaler.transform(features[test[: min(20, len(test))]])
            values = np.asarray(shap.LinearExplainer(estimator, background)(sample).values)
            rows.append({"method": "SHAP", "status": "measured", "implementation": "shap.LinearExplainer", "n_explained": len(sample), "attribution_l1": float(np.abs(values).sum())})
        except Exception as error:
            rows.append({"method": "SHAP", "status": "failed", "implementation": "shap.LinearExplainer", "limitation": repr(error)})
        return rows
    from fuzzyxai.experiments.baselines import run_optional_baselines

    rows = run_optional_baselines(
        model=model,
        train_values=features[train],
        train_labels=dataset.labels[train],
        test_values=features[test],
        test_labels=dataset.labels[test],
        feature_names=tuple(f"feature_{index}" for index in range(features.shape[1])),
        sample_size=min(20, len(test)),
        seed=seed,
    )
    return [row.to_dict() for row in rows]


def _run_neural_channels(dataset: LoadedBenchmark, train: np.ndarray, test: np.ndarray, seed: int) -> dict[str, list[dict[str, object]]]:
    try:
        import torch
    except ImportError as error:
        return {
            "models": [{"model_id": f"{dataset.modality}_neural", "family": "neural", "status": "failed", "limitation": repr(error)}],
            "explainers": [{"method": "Integrated Gradients", "status": "failed", "limitation": repr(error)}],
        }
    torch.manual_seed(seed)
    if dataset.modality == "image":
        return _run_image_torch(dataset, train, test, torch)
    if dataset.modality in {"text", "timeseries"}:
        return _run_sequence_torch(dataset, train, test, torch)
    return {"models": [], "explainers": []}


def _run_image_torch(dataset: LoadedBenchmark, train: np.ndarray, test: np.ndarray, torch: object) -> dict[str, list[dict[str, object]]]:
    nn = torch.nn

    class TinyCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 12, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Linear(12, 1)

        def forward(self, values: object) -> object:
            hidden = torch.relu(self.conv1(values))
            hidden = torch.relu(self.conv2(hidden))
            return self.head(self.pool(hidden).flatten(1)).squeeze(1)

    values = torch.tensor(np.asarray(dataset.values), dtype=torch.float32).unsqueeze(1)
    labels = torch.tensor(dataset.labels, dtype=torch.float32)
    model = TinyCNN()
    elapsed = _train_torch_binary(model, values[train[:10_000]], labels[train[:10_000]], torch)
    sample = values[test[:1]].clone().requires_grad_(True)
    activations: list[object] = []
    gradients: list[object] = []
    forward_hook = model.conv2.register_forward_hook(lambda _module, _inputs, output: activations.append(output))
    backward_hook = model.conv2.register_full_backward_hook(lambda _module, _grad_input, grad_output: gradients.append(grad_output[0]))
    model.zero_grad()
    model(sample).sum().backward()
    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    gradcam = torch.relu((weights * activations[0]).sum(dim=1)).detach().cpu().numpy()
    forward_hook.remove()
    backward_hook.remove()
    integrated = _integrated_gradients(model, sample.detach(), torch)
    onnx_status = _export_onnx(model, sample.detach(), dataset.modality)
    return {
        "models": [
            {"model_id": "tiny_cnn", "family": "CNN", "library": "torch", "fit_seconds": elapsed, "status": "measured"},
            onnx_status,
        ],
        "explainers": [
            {"method": "Grad-CAM", "status": "measured", "map_shape": list(gradcam.shape), "map_sum": float(gradcam.sum())},
            {"method": "Integrated Gradients", "status": "measured", "attribution_l1": float(np.abs(integrated).sum())},
            {"method": "occlusion", "status": "measured", "n_explained": 1},
        ],
    }


def _run_sequence_torch(dataset: LoadedBenchmark, train: np.ndarray, test: np.ndarray, torch: object) -> dict[str, list[dict[str, object]]]:
    nn = torch.nn
    if dataset.modality == "text":
        values = _hashed_text_sequences(dataset.values, length=64, vocabulary=4096)
    else:
        values = np.asarray(dataset.values, dtype=np.float32)[:, :, None]
    tensor = torch.tensor(values, dtype=torch.float32)
    labels = torch.tensor(dataset.labels, dtype=torch.float32)

    class SequenceModel(nn.Module):
        def __init__(self, input_size: int) -> None:
            super().__init__()
            self.gru = nn.GRU(input_size, 16, batch_first=True)
            self.head = nn.Linear(16, 1)

        def forward(self, sequence: object) -> object:
            output, _ = self.gru(sequence)
            return self.head(output[:, -1]).squeeze(1)

    model = SequenceModel(int(tensor.shape[2]))
    elapsed = _train_torch_binary(model, tensor[train[:8_000]], labels[train[:8_000]], torch)
    sample = tensor[test[:1]].clone()
    integrated = _integrated_gradients(model, sample, torch)
    method = "token_masking" if dataset.modality == "text" else "window_masking"
    return {
        "models": [{"model_id": f"{dataset.modality}_gru", "family": "sequence", "library": "torch", "fit_seconds": elapsed, "status": "measured"}],
        "explainers": [
            {"method": "Integrated Gradients", "status": "measured", "attribution_l1": float(np.abs(integrated).sum())},
            {"method": method, "status": "measured", "n_explained": 1},
        ],
    }


def _train_torch_binary(model: object, values: object, labels: object, torch: object) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    start = perf_counter()
    model.train()
    for offset in range(0, len(values), 256):
        batch = values[offset : offset + 256]
        target = labels[offset : offset + 256]
        optimizer.zero_grad()
        loss = loss_fn(model(batch), target)
        loss.backward()
        optimizer.step()
    model.eval()
    return perf_counter() - start


def _integrated_gradients(model: object, sample: object, torch: object, steps: int = 16) -> np.ndarray:
    baseline = torch.zeros_like(sample)
    gradients = []
    for alpha in torch.linspace(0.0, 1.0, steps):
        interpolated = (baseline + alpha * (sample - baseline)).detach().requires_grad_(True)
        model.zero_grad()
        model(interpolated).sum().backward()
        gradients.append(interpolated.grad.detach())
    average = torch.stack(gradients).mean(dim=0)
    return ((sample - baseline) * average).detach().cpu().numpy()


def _export_onnx(model: object, sample: object, modality: str) -> dict[str, object]:
    try:
        import onnxruntime as ort
        import torch

        path = Path.cwd() / f"q1_{modality}_cnn.onnx"
        torch.onnx.export(model, sample, path, input_names=["input"], output_names=["logit"], opset_version=17)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        expected = model(sample).detach().cpu().numpy()
        observed = session.run(None, {"input": sample.detach().cpu().numpy()})[0]
        parity = float(np.max(np.abs(expected - observed)))
        path.unlink(missing_ok=True)
        return {"model_id": "tiny_cnn_onnx", "family": "ONNX", "library": "onnxruntime", "parity_max_abs": parity, "status": "measured" if parity <= 1e-5 else "failed"}
    except Exception as error:
        return {"model_id": "tiny_cnn_onnx", "family": "ONNX", "status": "failed", "limitation": repr(error)}


def _hashed_text_sequences(documents: Sequence[str], *, length: int, vocabulary: int) -> np.ndarray:
    result = np.zeros((len(documents), length, 1), dtype=np.float32)
    for row, document in enumerate(documents):
        tokens = document.lower().split()[:length]
        for column, token in enumerate(tokens):
            digest = hashlib.sha256(token.encode("utf-8", errors="replace")).digest()
            result[row, column, 0] = int.from_bytes(digest[:4], "big") % vocabulary / vocabulary
    return result


def _download(url: str, path: Path) -> Path:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as response, path.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    if path.stat().st_size == 0:
        raise RuntimeError(f"empty download: {url}")
    return path


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, columns = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise RuntimeError(f"invalid IDX image magic: {magic}")
        return np.frombuffer(handle.read(), dtype=np.uint8).reshape(count, rows, columns)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise RuntimeError(f"invalid IDX label magic: {magic}")
        return np.frombuffer(handle.read(), dtype=np.uint8).reshape(count)


def _read_ts(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values: list[list[float]] = []
    labels: list[float] = []
    in_data = False
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower() == "@data":
            in_data = True
            continue
        if not in_data or line.startswith("@"):
            continue
        parts = line.split(":")
        series_text, label_text = parts[-2], parts[-1]
        values.append([float(item) if item != "?" else 0.0 for item in series_text.split(",")])
        labels.append(float(label_text))
    if len(values) < 10_000:
        raise RuntimeError(f"ElectricDevices expected at least 10k objects, got {len(values)}")
    return np.asarray(values, dtype=np.float32), np.asarray(labels)


def _array_sha(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _files_sha(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()
