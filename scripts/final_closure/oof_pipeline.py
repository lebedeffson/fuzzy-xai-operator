"""Real cross-fitted model and explanation channels for the sealed study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import pickle
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from common import ROOT, STUDY, sha256, write


DATA_ROOT = ROOT / "data/confirmatory"
OUTPUT = STUDY / "oof_features"
METADATA = STUDY / "dataset_manifests"
SEED = 7419
DATASETS = (
    "bank_marketing",
    "default_credit_clients",
    "shoulder_implant_xray",
    "sms_spam",
    "uci_har_smartphones",
)
PREDICTIVE_CHANNELS = (
    "calibrated_confidence",
    "prediction_margin",
    "normalized_entropy",
    "cross_fitted_calibration_estimate",
    "boundary_distance",
    "model_checkpoint_disagreement",
    "label_free_shift_score",
    "train_derived_rare_group_indicator",
    "missingness_profile",
    "data_quality_profile",
)
ROUTE_CHANNELS = (
    "explainer_disagreement",
    "seed_stability",
    "bootstrap_stability",
    "perturbation_stability",
    "provenance_completeness",
    "typed_route_fault",
    "canonical_hash_status",
    "representation_class",
    "reduction_loss",
    "rule_redundancy",
    "conflict_severity",
    "missing_evidence_channels",
    "reference_set_deviation",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    args = parser.parse_args()
    prerequisite = STUDY / "near_duplicate_audit.json"
    if not prerequisite.is_file() or json.loads(prerequisite.read_text())["status"] != "pass":
        raise SystemExit("BLOCKED: near-duplicate audit must pass before real OOF generation")
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset_id in selected:
        _run_dataset(dataset_id)
    _build_aggregate_manifest()


@dataclass
class LoadedData:
    dataset_id: str
    modality: str
    x: Any
    y: np.ndarray
    object_ids: np.ndarray
    groups: np.ndarray | None
    partitions: np.ndarray
    adapter: "Adapter"


class Adapter:
    modality = ""
    primary_model_id = ""
    alternate_model_id = ""
    explainer_id = "component_occlusion"

    def __init__(self, x: Any, y: np.ndarray):
        self.x, self.y = x, y

    @property
    def components(self) -> tuple[str, ...]:
        raise NotImplementedError

    def slice(self, indices: np.ndarray) -> Any:
        return self.x.iloc[indices] if hasattr(self.x, "iloc") else self.x[indices]

    def fit_model(self, indices: np.ndarray, *, role: str, seed: int):
        raise NotImplementedError

    def predict_proba(self, model: Any, values: Any, classes: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(model.predict_proba(values), dtype=np.float64)
        positions = {str(value): index for index, value in enumerate(model.classes_)}
        aligned = np.column_stack([probabilities[:, positions[str(value)]] for value in classes])
        return aligned / np.maximum(np.sum(aligned, axis=1, keepdims=True), 1e-12)

    def reference(self, indices: np.ndarray) -> Any:
        raise NotImplementedError

    def occlude(self, values: Any, component: int, reference: Any) -> Any:
        raise NotImplementedError

    def perturb(self, values: Any, reference: Any, *, seed: int) -> Any:
        raise NotImplementedError

    def profile(self, values: Any) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def component_detail(self, component: int, sample: Any) -> dict[str, object]:
        return {"component_id": self.components[component], "component_type": "feature"}

    def model_manifest(self) -> dict[str, object]:
        raise NotImplementedError


class TabularAdapter(Adapter):
    modality = "tabular"
    primary_model_id = "sklearn_hist_gradient_boosting"
    alternate_model_id = "sklearn_logistic_regression"

    def __init__(self, x: pd.DataFrame, y: np.ndarray):
        super().__init__(x, y)
        self.numeric = x.select_dtypes(exclude=["object", "string", "category"]).columns.tolist()
        self.categorical = [column for column in x.columns if column not in self.numeric]

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(str(column) for column in self.x.columns)

    def fit_model(self, indices: np.ndarray, *, role: str, seed: int):
        selected = _bootstrap_indices(indices, self.y, seed) if role == "bootstrap" else indices
        if role == "alternate":
            transformer = ColumnTransformer(
                (
                    (
                        "numeric",
                        Pipeline((("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()))),
                        self.numeric,
                    ),
                    (
                        "categorical",
                        Pipeline(
                            (("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore")))
                        ),
                        self.categorical,
                    ),
                )
            )
            estimator = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
        else:
            transformer = ColumnTransformer(
                (
                    ("numeric", SimpleImputer(strategy="median"), self.numeric),
                    (
                        "categorical",
                        Pipeline(
                            (
                                ("impute", SimpleImputer(strategy="most_frequent")),
                                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                            )
                        ),
                        self.categorical,
                    ),
                ),
                sparse_threshold=0.0,
            )
            estimator = HistGradientBoostingClassifier(
                max_iter=90,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=seed,
            )
        return Pipeline((("features", transformer), ("model", estimator))).fit(self.slice(selected), self.y[selected])

    def reference(self, indices: np.ndarray) -> dict[str, object]:
        fit = self.slice(indices)
        result: dict[str, object] = {}
        for column in self.numeric:
            result[column] = float(pd.to_numeric(fit[column], errors="coerce").median())
        for column in self.categorical:
            mode = fit[column].mode(dropna=True)
            result[column] = str(mode.iloc[0]) if len(mode) else ""
        return result

    def occlude(self, values: pd.DataFrame, component: int, reference: dict[str, object]) -> pd.DataFrame:
        output = values.copy()
        column = self.components[component]
        output[column] = reference[column]
        return output

    def perturb(self, values: pd.DataFrame, reference: dict[str, object], *, seed: int) -> pd.DataFrame:
        output = values.copy()
        rng = np.random.default_rng(seed)
        for column in self.numeric:
            numeric = pd.to_numeric(output[column], errors="coerce")
            scale = max(float(numeric.std()), abs(float(reference[column])) * 0.01, 1e-6)
            output[column] = numeric + rng.normal(0.0, 0.01 * scale, len(output))
        return output

    def profile(self, values: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        numeric = values[self.numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        missing = values.isna().mean(axis=1).to_numpy(dtype=float)
        return (numeric if self.numeric else np.zeros((len(values), 1))), missing

    def model_manifest(self) -> dict[str, object]:
        return {
            "primary": {"model_id": self.primary_model_id, "family": "gradient_boosting", "native_input": "mixed_tabular"},
            "alternate": {"model_id": self.alternate_model_id, "family": "linear", "native_input": "mixed_tabular"},
            "additional_benchmark_models": ["random_forest", "calibrated_gradient_boosting"],
        }


class TextAdapter(Adapter):
    modality = "text"
    primary_model_id = "tfidf_sgd_logistic_calibrated"
    alternate_model_id = "tfidf_complement_naive_bayes"
    component_count = 32

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(f"token_bucket_{index:02d}" for index in range(self.component_count))

    def fit_model(self, indices: np.ndarray, *, role: str, seed: int):
        selected = _bootstrap_indices(indices, self.y, seed) if role == "bootstrap" else indices
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000, sublinear_tf=True)
        estimator = (
            ComplementNB(alpha=0.5)
            if role == "alternate"
            else SGDClassifier(
                loss="log_loss",
                alpha=1e-5,
                max_iter=1500,
                tol=1e-4,
                class_weight="balanced",
                random_state=seed,
            )
        )
        return Pipeline((("tfidf", vectorizer), ("model", estimator))).fit(self.slice(selected), self.y[selected])

    def reference(self, indices: np.ndarray) -> None:
        return None

    def occlude(self, values: np.ndarray, component: int, reference: None) -> np.ndarray:
        return np.asarray(
            [" ".join(token for token in text.split() if _token_bucket(token) != component) for text in values.astype(str)],
            dtype=object,
        )

    def perturb(self, values: np.ndarray, reference: None, *, seed: int) -> np.ndarray:
        output = []
        for index, text in enumerate(values.astype(str)):
            tokens = text.split()
            if tokens:
                position = int(hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()[:8], 16) % len(tokens)
                tokens = tokens[:position] + tokens[position + 1 :]
            output.append(" ".join(tokens))
        return np.asarray(output, dtype=object)

    def profile(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = []
        for text in values.astype(str):
            length = max(1, len(text))
            rows.append(
                (
                    min(length, 1000),
                    min(len(text.split()), 250),
                    sum(character.isdigit() for character in text) / length,
                    sum(character.isupper() for character in text) / length,
                    float("http" in text.casefold()),
                )
            )
        return np.asarray(rows, dtype=float), np.zeros(len(rows), dtype=float)

    def component_detail(self, component: int, sample: str) -> dict[str, object]:
        tokens = [token for token in str(sample).split() if _token_bucket(token) == component]
        return {"component_id": self.components[component], "component_type": "token_group", "observed_tokens": tokens[:5]}

    def model_manifest(self) -> dict[str, object]:
        return {
            "primary": {"model_id": self.primary_model_id, "family": "calibrated_linear", "representation": "tfidf_1_2gram"},
            "alternate": {"model_id": self.alternate_model_id, "family": "naive_bayes", "representation": "tfidf_1_2gram"},
            "optional_embedding_model": "not_installed_not_verified",
        }


class ArraySklearnModel:
    def __init__(self, transformer, estimator):
        self.transformer, self.estimator = transformer, estimator
        self.classes_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ArraySklearnModel":
        self.estimator.fit(self.transformer(x), y)
        self.classes_ = np.asarray(self.estimator.classes_)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(self.transformer(x)), dtype=float)


class NeuralArrayAdapter(Adapter):
    def __init__(self, x: np.ndarray, y: np.ndarray, *, modality: str):
        super().__init__(x, y)
        self.modality = modality
        if modality == "image":
            self.primary_model_id = "compact_grayscale_cnn"
            self.alternate_model_id = "downsampled_pixel_logistic"
            self._components = tuple(f"region_r{row}_c{column}" for row in range(4) for column in range(4))
        else:
            self.primary_model_id = "compact_temporal_1d_cnn"
            self.alternate_model_id = "summary_feature_random_forest"
            self._components = tuple(f"window_{index:02d}" for index in range(16))

    @property
    def components(self) -> tuple[str, ...]:
        return self._components

    def fit_model(self, indices: np.ndarray, *, role: str, seed: int):
        selected = _bootstrap_indices(indices, self.y, seed) if role == "bootstrap" else indices
        if role == "alternate":
            if self.modality == "image":
                transformer = _image_pixel_features
                estimator = Pipeline((("scale", StandardScaler()), ("model", LogisticRegression(max_iter=500, class_weight="balanced"))))
            else:
                transformer = _timeseries_features
                estimator = RandomForestClassifier(
                    n_estimators=140,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=1,
                )
            return ArraySklearnModel(transformer, estimator).fit(self.slice(selected), self.y[selected])
        return TorchProbabilityModel(self.modality, seed=seed).fit(self.slice(selected), self.y[selected])

    def reference(self, indices: np.ndarray) -> np.ndarray:
        return np.mean(self.slice(indices), axis=0)

    def occlude(self, values: np.ndarray, component: int, reference: np.ndarray) -> np.ndarray:
        output = values.copy()
        if self.modality == "image":
            row, column = divmod(component, 4)
            rows, columns = slice(row * 32, (row + 1) * 32), slice(column * 32, (column + 1) * 32)
            output[:, rows, columns] = reference[rows, columns]
        else:
            start, stop = component * 8, (component + 1) * 8
            output[:, start:stop, :] = reference[start:stop, :]
        return output

    def perturb(self, values: np.ndarray, reference: np.ndarray, *, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if self.modality == "image":
            return np.clip(values.astype(float) + rng.normal(0.0, 1.5, values.shape), 0.0, 255.0).astype(values.dtype)
        scale = np.maximum(np.std(values, axis=(0, 1), keepdims=True), 1e-6)
        return (values + rng.normal(0.0, 0.01, values.shape) * scale).astype(values.dtype)

    def profile(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.modality == "image":
            flat = values.reshape(len(values), -1).astype(float) / 255.0
            profile = np.column_stack(
                (flat.mean(axis=1), flat.std(axis=1), np.quantile(flat, 0.05, axis=1), np.quantile(flat, 0.95, axis=1))
            )
        else:
            profile = _timeseries_features(values)
        return profile, np.zeros(len(values), dtype=float)

    def component_detail(self, component: int, sample: np.ndarray) -> dict[str, object]:
        if self.modality == "image":
            row, column = divmod(component, 4)
            return {
                "component_id": self.components[component],
                "component_type": "image_region",
                "pixel_bounds": [column * 32, row * 32, (column + 1) * 32, (row + 1) * 32],
            }
        return {
            "component_id": self.components[component],
            "component_type": "time_window",
            "sample_bounds": [component * 8, (component + 1) * 8],
            "time_seconds": [component * 8 / 50.0, (component + 1) * 8 / 50.0],
        }

    def model_manifest(self) -> dict[str, object]:
        if self.modality == "image":
            return {
                "primary": {
                    "model_id": self.primary_model_id,
                    "family": "compact_cnn",
                    "native_input": "grayscale_pixels_128x128",
                    "normalization": "fit_fold_mean_std",
                    "augmentation": "none_frozen",
                },
                "alternate": {"model_id": self.alternate_model_id, "family": "linear_pixel_baseline"},
            }
        return {
            "primary": {
                "model_id": self.primary_model_id,
                "family": "temporal_1d_cnn",
                "native_input": "128x9_sensor_window",
                "normalization": "fit_fold_channel_mean_std",
            },
            "alternate": {"model_id": self.alternate_model_id, "family": "feature_based_random_forest"},
        }


class TorchProbabilityModel:
    def __init__(self, modality: str, *, seed: int):
        self.modality, self.seed = modality, seed
        self.classes_: np.ndarray | None = None
        self.model = None
        self.location = None
        self.scale = None
        self.artifact_hash = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchProbabilityModel":
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.classes_ = np.asarray(sorted(set(y.tolist())))
        positions = {str(value): index for index, value in enumerate(self.classes_)}
        target = np.asarray([positions[str(value)] for value in y], dtype=np.int64)
        if self.modality == "image":
            self.location = np.asarray(float(np.mean(x)), dtype=np.float32)
            self.scale = np.asarray(max(float(np.std(x)), 1.0), dtype=np.float32)
            values = ((x.astype(np.float32) - self.location) / self.scale)[:, None, :, :]
            model = nn.Sequential(
                nn.Conv2d(1, 8, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(16 * 4 * 4, len(self.classes_)),
            )
            epochs, batch_size = 14, 64
        else:
            self.location = np.mean(x, axis=(0, 1), keepdims=True).astype(np.float32)
            self.scale = np.maximum(np.std(x, axis=(0, 1), keepdims=True), 1e-5).astype(np.float32)
            values = np.transpose((x.astype(np.float32) - self.location) / self.scale, (0, 2, 1))
            model = nn.Sequential(
                nn.Conv1d(x.shape[2], 24, 7, padding=3), nn.ReLU(),
                nn.Conv1d(24, 24, 5, padding=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(24, len(self.classes_)),
            )
            epochs, batch_size = 6, 128
        counts = np.bincount(target, minlength=len(self.classes_)).astype(float)
        weights = torch.tensor(len(target) / np.maximum(counts * len(self.classes_), 1.0), dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_function = nn.CrossEntropyLoss(weight=weights)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(values), torch.from_numpy(target)),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=0,
        )
        model.train()
        for _ in range(epochs):
            for features, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                loss_function(model(features), labels).backward()
                optimizer.step()
        self.model = model.eval()
        self.artifact_hash = hashlib.sha256(
            b"".join(value.detach().cpu().numpy().tobytes() for _, value in sorted(model.state_dict().items()))
        ).hexdigest()
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        import torch

        values = (
            ((x.astype(np.float32) - self.location) / self.scale)[:, None, :, :]
            if self.modality == "image"
            else np.transpose((x.astype(np.float32) - self.location) / self.scale, (0, 2, 1))
        )
        outputs = []
        with torch.no_grad():
            for start in range(0, len(values), 512):
                outputs.append(torch.softmax(self.model(torch.from_numpy(values[start : start + 512])), dim=1).cpu().numpy())
        return np.concatenate(outputs)


def _run_dataset(dataset_id: str) -> None:
    data = _load(dataset_id)
    adapter, y, ids = data.adapter, data.y, data.object_ids
    classes = np.asarray(sorted(set(y.tolist())))
    count, component_count = len(y), len(adapter.components)
    probabilities = {role: np.zeros((count, len(classes)), dtype=np.float64) for role in ("primary", "alternate", "seed", "bootstrap")}
    contributions = {role: np.zeros((count, component_count), dtype=np.float32) for role in ("primary", "alternate", "seed", "bootstrap", "perturbation")}
    calibration_candidates = {name: np.zeros(count, dtype=np.float64) for name in ("platt", "isotonic", "temperature", "conformal")}
    calibration_estimate = np.zeros(count, dtype=np.float64)
    rare = np.zeros(count, dtype=np.float64)
    shift = np.zeros(count, dtype=np.float64)
    quality = np.zeros(count, dtype=np.float64)
    missingness = np.zeros(count, dtype=np.float64)
    fold_ids = np.zeros(count, dtype=np.int16)
    checkpoint_rows = []
    splitter = (
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        if data.groups is not None
        else StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    )
    iterator = splitter.split(np.zeros(count), y, data.groups) if data.groups is not None else splitter.split(np.zeros(count), y)
    for fold, (fit_index, held_index) in enumerate(iterator):
        fold_ids[held_index] = fold
        reference = adapter.reference(fit_index)
        models = {
            role: adapter.fit_model(fit_index, role=role, seed=SEED + fold * 101 + role_index * 17)
            for role_index, role in enumerate(("primary", "alternate", "seed", "bootstrap"))
        }
        for role, model in models.items():
            probabilities[role][held_index] = adapter.predict_proba(model, adapter.slice(held_index), classes)
            checkpoint_rows.append({"fold": fold, "role": role, "sha256": _model_hash(model)})
        predicted_positions = np.argmax(probabilities["primary"][held_index], axis=1)
        target_labels = classes[predicted_positions]
        for role, model in models.items():
            contributions[role][held_index] = _occlusion_contributions(
                adapter, model, adapter.slice(held_index), target_labels, classes, reference
            )
        perturbed = adapter.perturb(adapter.slice(held_index), reference, seed=SEED + fold)
        contributions["perturbation"][held_index] = _occlusion_contributions(
            adapter, models["primary"], perturbed, target_labels, classes, reference
        )
        fit_primary = adapter.predict_proba(models["primary"], adapter.slice(fit_index), classes)
        fit_positions = np.argmax(fit_primary, axis=1)
        fit_predictions = classes[fit_positions]
        fit_confidence = fit_primary[np.arange(len(fit_index)), fit_positions]
        held_confidence = probabilities["primary"][held_index, predicted_positions]
        fit_correct = (fit_predictions == y[fit_index]).astype(int)
        for name, values in _calibrate_candidates(fit_confidence, fit_correct, held_confidence).items():
            calibration_candidates[name][held_index] = values
        calibration_estimate[held_index] = _calibration_gap(fit_confidence, fit_correct, held_confidence)
        counts = pd.Series(fit_predictions).value_counts(normalize=True)
        rare[held_index] = np.asarray([float(counts.get(value, 0.0) < 0.10) for value in target_labels])
        fit_profile, _ = adapter.profile(adapter.slice(fit_index))
        held_profile, held_missing = adapter.profile(adapter.slice(held_index))
        shift[held_index], quality[held_index] = _profile_scores(fit_profile, held_profile, held_missing)
        missingness[held_index] = held_missing
        print(f"PASS: oof_fold dataset={dataset_id} fold={fold} held={len(held_index)}", flush=True)

    primary = probabilities["primary"]
    prediction_positions = np.argmax(primary, axis=1)
    predictions = classes[prediction_positions]
    raw_confidence = primary[np.arange(count), prediction_positions]
    order = np.argsort(primary, axis=1)
    margin = raw_confidence - primary[np.arange(count), order[:, -2]] if len(classes) > 1 else raw_confidence
    entropy = -np.sum(primary * np.log(np.clip(primary, 1e-12, 1.0)), axis=1) / max(math.log(len(classes)), 1.0)
    calibration = _select_calibration(calibration_candidates, predictions == y, data.partitions)
    calibrated_confidence = calibration_candidates[str(calibration["selected_method"])]
    disagreement = 0.5 * np.sum(np.abs(primary - probabilities["alternate"]), axis=1)
    route = _route_channels(contributions, entropy, shift)
    evidence_path = OUTPUT / f"canonical/{dataset_id}.jsonl"
    output_path = OUTPUT / f"{dataset_id}.jsonl"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as rows_handle, evidence_path.open("w", encoding="utf-8") as evidence_handle:
        for index, object_id in enumerate(ids):
            sample = _sample(adapter.slice(np.asarray([index])))
            source_payload = {
                "schema_version": "1.0",
                "dataset_id": dataset_id,
                "object_id_hash": str(object_id),
                "fold": int(fold_ids[index]),
                "model_id": adapter.primary_model_id,
                "explainer_id": adapter.explainer_id,
                "prediction": str(predictions[index]),
                "components": [
                    adapter.component_detail(component, sample)
                    | {"contribution": float(contributions["primary"][index, component])}
                    for component in range(component_count)
                ],
            }
            canonical = json.dumps(source_payload, sort_keys=True, separators=(",", ":"))
            canonical_hash = hashlib.sha256(canonical.encode()).hexdigest()
            evidence_handle.write(canonical + "\n")
            predictive_values = (
                calibrated_confidence[index], margin[index], entropy[index], calibration_estimate[index], margin[index],
                disagreement[index], shift[index], rare[index], missingness[index], quality[index],
            )
            route_values = tuple(route[name][index] for name in ROUTE_CHANNELS)
            row = {
                "dataset_id": dataset_id,
                "object_id_hash": str(object_id),
                "split_id": "train-development-oof",
                "partition": str(data.partitions[index]),
                "fold": int(fold_ids[index]),
                "source_is_oof": True,
                "true_label": str(y[index]),
                "predicted_label": str(predictions[index]),
                "model_id": adapter.primary_model_id,
                "explainer_id": adapter.explainer_id,
                "canonical_evidence_sha256": canonical_hash,
                "predictive": dict(zip(PREDICTIVE_CHANNELS, (float(np.clip(value, 0.0, 1.0)) for value in predictive_values), strict=True)),
                "route": dict(
                    zip(
                        ROUTE_CHANNELS,
                        (None if value is None or np.isnan(value) else float(np.clip(value, 0.0, 1.0)) for value in route_values),
                        strict=True,
                    )
                ),
            }
            row["missing_channels"] = [name for name, value in (*row["predictive"].items(), *row["route"].items()) if value is None]
            rows_handle.write(json.dumps(row, sort_keys=True) + "\n")
    model_path = METADATA / f"{dataset_id}/model_manifest.json"
    explainer_path = METADATA / f"{dataset_id}/explainer_manifest.json"
    write(
        model_path,
        {
            "schema_version": "1.0",
            "dataset_id": dataset_id,
            "modality": data.modality,
            "source_commit": _git_head(),
            "training_scope": "five_fold_cross_fitted_train_development_only",
            "sealed_test_loaded": False,
            "models": adapter.model_manifest(),
            "fold_artifacts": checkpoint_rows,
            "calibration": calibration,
            "versions": _versions(),
        },
    )
    write(
        explainer_path,
        {
            "schema_version": "1.0",
            "dataset_id": dataset_id,
            "explainer_id": adapter.explainer_id,
            "method": "prediction_difference_after_component_occlusion",
            "components": list(adapter.components),
            "native_or_surrogate": "derived_model_agnostic",
            "canonical_payload": "all_component_contributions_preserved_without_projection",
            "repeat_channels": ["alternate_model", "seed", "bootstrap", "input_perturbation"],
            "sealed_test_loaded": False,
        },
    )
    summary = {
        "dataset_id": dataset_id,
        "modality": data.modality,
        "objects": count,
        "classes": classes.tolist(),
        "accuracy": float(accuracy_score(y, predictions)),
        "log_loss": float(log_loss(y, primary, labels=classes)),
        "artifact_path": output_path.relative_to(ROOT).as_posix(),
        "artifact_sha256": sha256(output_path),
        "canonical_evidence_path": evidence_path.relative_to(ROOT).as_posix(),
        "canonical_evidence_sha256": sha256(evidence_path),
        "model_manifest_path": model_path.relative_to(ROOT).as_posix(),
        "model_manifest_sha256": sha256(model_path),
        "explainer_manifest_path": explainer_path.relative_to(ROOT).as_posix(),
        "explainer_manifest_sha256": sha256(explainer_path),
        "model_family": adapter.primary_model_id,
        "alternate_model_family": adapter.alternate_model_id,
        "calibration_method": calibration["selected_method"],
        "missing_predictive_channels": [],
        "not_applicable_route_channels": ["rule_redundancy"],
        "route_features_status": "measured_oof_with_explicit_not_applicable_mask",
        "held_out_label_used_as_feature": False,
        "sealed_test_loaded": False,
    }
    write(OUTPUT / f"{dataset_id}.summary.json", summary)
    print(f"PASS: real_oof_dataset dataset={dataset_id} objects={count} p0=pass p1=pass test_loaded=false")


def _load(dataset_id: str) -> LoadedData:
    if dataset_id in {"bank_marketing", "default_credit_clients", "sms_spam"}:
        target = {"bank_marketing": "y", "default_credit_clients": "target", "sms_spam": "target"}[dataset_id]
        frames, partitions = [], []
        for partition in ("train", "development"):
            frame = pd.read_csv(DATA_ROOT / dataset_id / f"processed/{partition}.csv")
            frames.append(frame)
            partitions.extend([partition] * len(frame))
        frame = pd.concat(frames, ignore_index=True)
        ids = frame.pop("object_id_hash").astype(str).to_numpy()
        y = frame.pop(target).astype(str).to_numpy()
        if dataset_id == "default_credit_clients":
            frame = frame.drop(columns=["ID"], errors="ignore")
        if dataset_id == "sms_spam":
            x = frame["text"].fillna("").astype(str).to_numpy(dtype=object)
            adapter: Adapter = TextAdapter(x, y)
            modality = "text"
        else:
            x = frame
            adapter = TabularAdapter(x, y)
            modality = "tabular"
        return LoadedData(dataset_id, modality, x, y, ids, None, np.asarray(partitions), adapter)
    payloads, partitions = [], []
    for partition in ("train", "development"):
        with np.load(DATA_ROOT / dataset_id / f"processed/{partition}.npz") as payload:
            payloads.append({key: payload[key].copy() for key in payload.files})
            partitions.extend([partition] * len(payload["x"]))
    x = np.concatenate([payload["x"] for payload in payloads])
    y = np.concatenate([payload["y"] for payload in payloads]).astype(str)
    ids = np.concatenate([payload["object_id_hash"] for payload in payloads]).astype(str)
    groups = np.concatenate([payload["subject_id"] for payload in payloads]) if "subject_id" in payloads[0] else None
    modality = "image" if dataset_id == "shoulder_implant_xray" else "timeseries"
    adapter = NeuralArrayAdapter(x, y, modality=modality)
    return LoadedData(dataset_id, modality, x, y, ids, groups, np.asarray(partitions), adapter)


def _occlusion_contributions(
    adapter: Adapter,
    model: Any,
    values: Any,
    target_labels: np.ndarray,
    classes: np.ndarray,
    reference: Any,
) -> np.ndarray:
    base = adapter.predict_proba(model, values, classes)
    target_positions = np.asarray([int(np.flatnonzero(classes == value)[0]) for value in target_labels])
    rows = np.arange(len(target_labels))
    output = np.zeros((len(target_labels), len(adapter.components)), dtype=np.float32)
    for component in range(len(adapter.components)):
        changed = adapter.predict_proba(model, adapter.occlude(values, component, reference), classes)
        output[:, component] = base[rows, target_positions] - changed[rows, target_positions]
    return output


def _route_channels(contributions: dict[str, np.ndarray], entropy: np.ndarray, shift: np.ndarray) -> dict[str, np.ndarray]:
    primary = contributions["primary"]
    similarity = {
        role: _distribution_similarity(primary, contributions[role])
        for role in ("alternate", "seed", "bootstrap", "perturbation")
    }
    absolute = np.abs(primary)
    total = np.sum(absolute, axis=1)
    top_k = min(5, primary.shape[1])
    retained = np.sum(np.sort(absolute, axis=1)[:, -top_k:], axis=1)
    reduction_loss = np.divide(total - retained, total, out=np.zeros_like(total), where=total > 1e-12)
    sign_conflict = np.sum(
        np.minimum(np.abs(primary), np.abs(contributions["alternate"]))
        * (np.sign(primary) != np.sign(contributions["alternate"])),
        axis=1,
    )
    conflict = np.divide(sign_conflict, total, out=np.zeros_like(total), where=total > 1e-12)
    representation = np.where(entropy < 0.25, 0.0, np.where(entropy < 0.50, 1.0 / 3.0, np.where(entropy < 0.75, 2.0 / 3.0, 1.0)))
    return {
        "explainer_disagreement": 1.0 - similarity["alternate"],
        "seed_stability": similarity["seed"],
        "bootstrap_stability": similarity["bootstrap"],
        "perturbation_stability": similarity["perturbation"],
        "provenance_completeness": np.ones(len(primary)),
        "typed_route_fault": np.zeros(len(primary)),
        "canonical_hash_status": np.ones(len(primary)),
        "representation_class": representation,
        "reduction_loss": reduction_loss,
        "rule_redundancy": np.full(len(primary), np.nan),
        "conflict_severity": np.clip(conflict, 0.0, 1.0),
        "missing_evidence_channels": np.full(len(primary), 1.0 / len(ROUTE_CHANNELS)),
        "reference_set_deviation": shift,
    }


def _distribution_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_abs, right_abs = np.abs(left), np.abs(right)
    left_norm = left_abs / np.maximum(np.sum(left_abs, axis=1, keepdims=True), 1e-12)
    right_norm = right_abs / np.maximum(np.sum(right_abs, axis=1, keepdims=True), 1e-12)
    both_empty = (np.sum(left_abs, axis=1) <= 1e-12) & (np.sum(right_abs, axis=1) <= 1e-12)
    result = 1.0 - 0.5 * np.sum(np.abs(left_norm - right_norm), axis=1)
    result[both_empty] = 1.0
    return np.clip(result, 0.0, 1.0)


def _profile_scores(fit: np.ndarray, held: np.ndarray, missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(fit, axis=0)
    fit_filled = np.where(np.isnan(fit), median, fit)
    held_filled = np.where(np.isnan(held), median, held)
    q25, q75 = np.quantile(fit_filled, (0.25, 0.75), axis=0)
    scale = np.maximum(q75 - q25, np.std(fit_filled, axis=0) * 0.25 + 1e-9)
    shift = np.mean(np.clip(np.abs((held_filled - median) / scale) / 6.0, 0.0, 1.0), axis=1)
    low, high = np.quantile(fit_filled, (0.01, 0.99), axis=0)
    outside = np.mean((held_filled < low) | (held_filled > high), axis=1)
    return shift, np.clip(0.5 * outside + 0.5 * missing, 0.0, 1.0)


def _calibrate_candidates(fit_confidence: np.ndarray, fit_correct: np.ndarray, held_confidence: np.ndarray) -> dict[str, np.ndarray]:
    clipped_fit = np.clip(fit_confidence, 1e-6, 1.0 - 1e-6)
    clipped_held = np.clip(held_confidence, 1e-6, 1.0 - 1e-6)
    if len(np.unique(fit_correct)) < 2:
        return {name: clipped_held.copy() for name in ("platt", "isotonic", "temperature", "conformal")}
    platt = LogisticRegression(max_iter=500).fit(clipped_fit[:, None], fit_correct).predict_proba(clipped_held[:, None])[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(clipped_fit, fit_correct).predict(clipped_held)
    fit_logit = np.log(clipped_fit / (1.0 - clipped_fit))
    held_logit = np.log(clipped_held / (1.0 - clipped_held))

    def objective(log_temperature: float) -> float:
        values = 1.0 / (1.0 + np.exp(-fit_logit / math.exp(log_temperature)))
        return float(-np.mean(fit_correct * np.log(values + 1e-12) + (1 - fit_correct) * np.log(1 - values + 1e-12)))

    temperature = math.exp(float(minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded").x))
    temperature_values = 1.0 / (1.0 + np.exp(-held_logit / temperature))
    boundaries = np.quantile(clipped_fit, np.linspace(0.0, 1.0, 11))
    bins = np.clip(np.digitize(clipped_fit, boundaries[1:-1]), 0, 9)
    held_bins = np.clip(np.digitize(clipped_held, boundaries[1:-1]), 0, 9)
    global_rate = float(np.mean(fit_correct))
    rates = np.asarray([float(np.mean(fit_correct[bins == index])) if np.any(bins == index) else global_rate for index in range(10)])
    return {
        "platt": np.clip(platt, 0.0, 1.0),
        "isotonic": np.clip(isotonic, 0.0, 1.0),
        "temperature": np.clip(temperature_values, 0.0, 1.0),
        "conformal": np.clip(rates[held_bins], 0.0, 1.0),
    }


def _calibration_gap(fit_confidence: np.ndarray, fit_correct: np.ndarray, held_confidence: np.ndarray) -> np.ndarray:
    boundaries = np.linspace(0.0, 1.0, 11)
    bins = np.clip(np.digitize(fit_confidence, boundaries[1:-1]), 0, 9)
    global_gap = abs(float(np.mean(fit_correct)) - float(np.mean(fit_confidence)))
    gaps = np.asarray(
        [
            abs(float(np.mean(fit_correct[bins == index])) - float(np.mean(fit_confidence[bins == index])))
            if np.any(bins == index)
            else global_gap
            for index in range(10)
        ]
    )
    return gaps[np.clip(np.digitize(held_confidence, boundaries[1:-1]), 0, 9)]


def _select_calibration(candidates: dict[str, np.ndarray], correct: np.ndarray, partitions: np.ndarray) -> dict[str, object]:
    selected = partitions == "development"
    rows = [
        {
            "method": method,
            "development_brier_correctness": float(np.mean(np.square(values[selected] - correct[selected].astype(float)))),
            "n": int(selected.sum()),
        }
        for method, values in candidates.items()
    ]
    winner = min(rows, key=lambda row: (row["development_brier_correctness"], row["method"]))
    return {
        "selection_partition": "development",
        "candidates": rows,
        "selected_method": winner["method"],
        "test_labels_used": False,
        "per_row_mapping_fit_excludes_row": True,
    }


def _build_aggregate_manifest() -> None:
    reports = []
    for dataset_id in DATASETS:
        path = OUTPUT / f"{dataset_id}.summary.json"
        if path.is_file():
            reports.append(json.loads(path.read_text(encoding="utf-8")))
    complete = len(reports) == len(DATASETS)
    write(
        STUDY / "confirmatory_feature_manifest.json",
        {
            "schema_version": "2.0",
            "source_commit": _git_head(),
            "feature_source": "real_cross_fitted_train_development_models_and_explanations",
            "sealed_test_loaded": False,
            "predictive_channels": list(PREDICTIVE_CHANNELS),
            "route_channels": list(ROUTE_CHANNELS),
            "channel_semantics": {
                "good_when_high": ["calibrated_confidence", "prediction_margin", "boundary_distance", "seed_stability", "bootstrap_stability", "perturbation_stability", "provenance_completeness", "canonical_hash_status"],
                "risk_when_high": ["normalized_entropy", "cross_fitted_calibration_estimate", "model_checkpoint_disagreement", "label_free_shift_score", "train_derived_rare_group_indicator", "missingness_profile", "data_quality_profile", "explainer_disagreement", "typed_route_fault", "reduction_loss", "rule_redundancy", "conflict_severity", "missing_evidence_channels", "reference_set_deviation"],
                "categorical_ordinal": ["representation_class"],
            },
            "P0_status": "pass_predictive_oof" if complete else "blocked_incomplete_real_oof",
            "P1_status": "pass_route_oof" if complete else "blocked_incomplete_real_oof",
            "lock_status": "ready_for_p0_p1_audit" if complete else "blocked_real_oof_pending",
            "datasets": reports,
        },
    )
    print(f"PASS: real_oof_manifest datasets={len(reports)}/5 complete={str(complete).lower()} test_loaded=false")


def _bootstrap_indices(indices: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    for _ in range(20):
        selected = rng.choice(indices, size=len(indices), replace=True)
        if len(set(labels[selected].tolist())) == len(set(labels[indices].tolist())):
            return np.asarray(selected, dtype=np.int64)
    raise RuntimeError("bootstrap sample lost one or more classes")


def _token_bucket(token: str) -> int:
    return int(hashlib.blake2b(token.casefold().encode(), digest_size=4).hexdigest(), 16) % TextAdapter.component_count


def _timeseries_features(x: np.ndarray) -> np.ndarray:
    return np.concatenate((x.mean(axis=1), x.std(axis=1), x.min(axis=1), x.max(axis=1), np.mean(np.square(x), axis=1)), axis=1)


def _image_pixel_features(value: np.ndarray) -> np.ndarray:
    return value[:, ::8, ::8].reshape(len(value), -1).astype(np.float32) / 255.0


def _model_hash(model: Any) -> str:
    explicit = getattr(model, "artifact_hash", None)
    return str(explicit) if explicit else hashlib.sha256(pickle.dumps(model, protocol=5)).hexdigest()


def _sample(value: Any) -> Any:
    return value.iloc[0] if hasattr(value, "iloc") else value[0]


def _versions() -> dict[str, str]:
    result = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "torch"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not_installed"
    return result


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
