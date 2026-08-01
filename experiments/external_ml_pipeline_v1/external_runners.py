from __future__ import annotations

import hashlib
import json
import pickle
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import shap
import sklearn
from lime.lime_tabular import LimeTabularExplainer
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 1729


@dataclass(frozen=True)
class ExternalSpec:
    pipeline_id: str
    repository_url: str
    repository_commit: str
    license: str
    source_files: tuple[str, ...]
    task_type: str
    dataset_source: str
    explainer: str


SPECS = (
    ExternalSpec(
        "ext1-sklearn-column-transformer",
        "https://github.com/scikit-learn/scikit-learn",
        "5799d3eac08bda44fbce3309e641cbf98c5d312a",
        "BSD-3-Clause",
        ("examples/compose/plot_column_transformer_mixed_types.py",),
        "BINARY_CLASSIFICATION",
        "sklearn.datasets.load_breast_cancer",
        "shap.LinearExplainer",
    ),
    ExternalSpec(
        "ext2-shap-tree-explainer",
        "https://github.com/shap/shap",
        "7716688e0f314d73ed2f90474e911c85bc53c851",
        "MIT",
        ("tests/explainers/test_tree.py",),
        "MULTICLASS_CLASSIFICATION",
        "sklearn.datasets.load_digits",
        "shap.TreeExplainer",
    ),
    ExternalSpec(
        "ext3-mlflow-elasticnet",
        "https://github.com/mlflow/mlflow",
        "6e168e696062942ee34410048ce17cf24b5118bc",
        "Apache-2.0",
        ("examples/sklearn_elasticnet_wine/train.py",),
        "REGRESSION",
        "sklearn.datasets.load_diabetes",
        "shap.LinearExplainer",
    ),
    ExternalSpec(
        "ext4-lime-tabular",
        "https://github.com/marcotcr/lime",
        "fd7eb2e6f760619c29fca0187c07b82157601b32",
        "BSD-3-Clause",
        ("lime/lime_tabular.py",),
        "MULTICLASS_CLASSIFICATION",
        "sklearn.datasets.load_wine",
        "lime.LimeTabularExplainer",
    ),
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split(frame: pd.DataFrame, target: np.ndarray, *, stratify: bool) -> tuple[Any, ...]:
    indices = np.arange(len(frame))
    labels = target if stratify else None
    train, test = train_test_split(indices, test_size=0.25, random_state=SEED, stratify=labels)
    return train, test, frame.iloc[train].copy(), frame.iloc[test].copy(), target[train], target[test]


def _shap_linear(model: Any, train: np.ndarray, sample: np.ndarray, selected: int | None) -> dict[str, Any]:
    explainer = shap.LinearExplainer(model, train[: min(128, len(train))])
    raw = explainer(sample)
    values = np.asarray(raw.values, dtype=float)
    bases = np.asarray(raw.base_values, dtype=float)
    if values.ndim == 3:
        output = int(selected or 0)
        contributions = values[0, :, output]
        base = float(bases.reshape(-1)[output])
    else:
        contributions = values[0]
        base = float(bases.reshape(-1)[0])
    return {
        "explainer_id": "shap.LinearExplainer",
        "explainer_version": shap.__version__,
        "base_value": base,
        "attributions": contributions.tolist(),
        "selected_output": selected,
    }


def _shap_tree(model: Any, sample: np.ndarray, selected: int) -> dict[str, Any]:
    raw = shap.TreeExplainer(model)(sample)
    values = np.asarray(raw.values, dtype=float)
    bases = np.asarray(raw.base_values, dtype=float)
    if values.ndim == 3:
        contributions = values[0, :, selected]
        base = float(bases.reshape(-1)[selected])
    else:
        contributions = values[0]
        base = float(bases.reshape(-1)[0])
    return {
        "explainer_id": "shap.TreeExplainer",
        "explainer_version": shap.__version__,
        "base_value": base,
        "attributions": contributions.tolist(),
        "selected_output": selected,
    }


def _prepare_data(spec: ExternalSpec) -> tuple[pd.DataFrame, np.ndarray, Any]:
    if spec.pipeline_id.startswith("ext1"):
        loaded = load_breast_cancer(as_frame=True)
        frame = loaded.data.iloc[:, :8].copy()
        frame["radius_band"] = pd.qcut(frame.iloc[:, 0], 4, labels=["q1", "q2", "q3", "q4"]).astype(str)
        frame["texture_band"] = pd.qcut(frame.iloc[:, 1], 3, labels=["low", "mid", "high"]).astype(str)
        return frame, np.asarray(loaded.target), loaded
    if spec.pipeline_id.startswith("ext2"):
        loaded = load_digits(as_frame=True)
        return loaded.data.iloc[:600].copy(), np.asarray(loaded.target[:600]), loaded
    if spec.pipeline_id.startswith("ext3"):
        loaded = load_diabetes(as_frame=True)
        return loaded.data.copy(), np.asarray(loaded.target, dtype=float), loaded
    loaded = load_wine(as_frame=True)
    return loaded.data.copy(), np.asarray(loaded.target), loaded


def run_external_pipeline(spec: ExternalSpec, root: Path, *, retrained: bool = False) -> Path:
    """Execute a public-library pipeline without importing FuzzyXAI."""

    variant = "retrained" if retrained else "baseline"
    output = root / spec.pipeline_id / variant
    output.mkdir(parents=True, exist_ok=True)
    frame, target, _ = _prepare_data(spec)
    train_i, test_i, train_x, test_x, train_y, _test_y = _split(frame, target, stratify="CLASSIFICATION" in spec.task_type)
    if spec.pipeline_id.startswith("ext1"):
        numeric = list(train_x.select_dtypes(include=[np.number]).columns)
        categorical = [name for name in train_x.columns if name not in numeric]
        preprocessor: Any = ColumnTransformer(
            [
                ("numeric", StandardScaler(), numeric),
                ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ]
        )
        model: Any = LogisticRegression(max_iter=400, random_state=SEED, C=0.8 if retrained else 1.0)
    elif spec.pipeline_id.startswith("ext2"):
        preprocessor = StandardScaler()
        model = RandomForestClassifier(n_estimators=24, max_depth=8, random_state=SEED + int(retrained))
    elif spec.pipeline_id.startswith("ext3"):
        preprocessor = StandardScaler()
        model = ElasticNet(alpha=0.08 if retrained else 0.1, l1_ratio=0.5, random_state=SEED, max_iter=3000)
    else:
        preprocessor = StandardScaler()
        model = RandomForestClassifier(n_estimators=24, max_depth=6, random_state=SEED + int(retrained))
    transformed_train = np.asarray(preprocessor.fit_transform(train_x), dtype=float)
    transformed_test = np.asarray(preprocessor.transform(test_x), dtype=float)
    model.fit(transformed_train, train_y)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    artifact_path = output / "model.pkl"
    artifact_path.write_bytes(pickle.dumps(pipeline, protocol=5))
    object_id = f"{spec.pipeline_id}:row:{int(test_i[0])}"
    sample = transformed_test[:1]
    predicted = model.predict(sample)[0]
    if hasattr(model, "predict_proba"):
        selected = int(np.where(model.classes_ == predicted)[0][0])
        model_output = float(model.predict_proba(sample)[0, selected])
    else:
        selected = None
        model_output = float(predicted)
    if spec.pipeline_id.startswith("ext2"):
        explanation = _shap_tree(model, sample, int(selected or 0))
    elif spec.pipeline_id.startswith("ext4"):
        lime_explainer = LimeTabularExplainer(
            transformed_train,
            feature_names=[str(name) for name in frame.columns],
            class_names=[str(item) for item in model.classes_],
            random_state=SEED,
            mode="classification",
        )
        local = lime_explainer.explain_instance(sample[0], model.predict_proba, num_features=min(10, sample.shape[1]))
        explanation = {
            "explainer_id": "lime.LimeTabularExplainer",
            "explainer_version": "0.2.0.1",
            "base_value": float(local.intercept[int(selected or 0)]),
            "attributions": [[str(key), float(value)] for key, value in local.as_list(label=int(selected or 0))],
            "selected_output": selected,
        }
    else:
        explanation = _shap_linear(model, transformed_train, sample, selected)
    feature_names = (
        [str(item) for item in preprocessor.get_feature_names_out()]
        if hasattr(preprocessor, "get_feature_names_out")
        else [str(item) for item in frame.columns]
    )
    model_sha = _sha(artifact_path)
    dataset_payload = {
        "dataset_source": spec.dataset_source,
        "dataset_sha256": _digest({"features": frame.astype(str).values.tolist(), "target": target.tolist()}),
        "row_count": len(frame),
        "column_count": frame.shape[1],
        "feature_names": [str(item) for item in frame.columns],
        "target_name": "target",
        "target_in_features": False,
    }
    split_payload = {
        "random_state": SEED,
        "train_ids": [int(item) for item in train_i],
        "test_ids": [int(item) for item in test_i],
        "intersection_count": 0,
        "split_sha256": _digest({"train": train_i.tolist(), "test": test_i.tolist()}),
    }
    preprocessor_bytes = pickle.dumps(preprocessor, protocol=5)
    preprocessor_payload = {
        "class_name": type(preprocessor).__name__,
        "artifact_sha256": hashlib.sha256(preprocessor_bytes).hexdigest(),
        "fit_row_ids_sha256": _digest(train_i.tolist()),
        "expected_train_row_ids_sha256": _digest(train_i.tolist()),
        "fit_row_count": len(train_i),
        "feature_names_in": [str(item) for item in frame.columns],
        "feature_names_out": feature_names,
        "output_feature_count": len(feature_names),
    }
    model_payload = {
        "model_class": type(model).__name__,
        "model_version": variant,
        "artifact_path": "model.pkl",
        "artifact_sha256": model_sha,
        "registered_artifact_sha256": model_sha,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
    }
    prediction_payload = {
        "object_id": object_id,
        "run_id": f"external:{spec.pipeline_id}:{variant}",
        "model_sha256": model_sha,
        "selected_output": selected,
        "prediction": predicted.item() if hasattr(predicted, "item") else predicted,
        "model_output": model_output,
        "input_sha256": _digest(sample.tolist()),
    }
    explanation_payload = {
        **explanation,
        "object_id": object_id,
        "run_id": prediction_payload["run_id"],
        "model_sha256": model_sha,
        "model_version": variant,
        "feature_names": feature_names,
        "artifact_sha256": _digest(explanation),
        "source_uri": f"{spec.repository_url}/blob/{spec.repository_commit}/{spec.source_files[0]}",
    }
    run_payload = {
        "run_id": prediction_payload["run_id"],
        "variant": variant,
        "registered_model_version": variant,
        "model_sha256": model_sha,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "shap_version": shap.__version__,
        "mlflow_version": mlflow.__version__,
    }
    pipeline_payload = {
        "pipeline_id": spec.pipeline_id,
        "repository_url": spec.repository_url,
        "repository_commit": spec.repository_commit,
        "license": spec.license,
        "source_files": spec.source_files,
        "task_type": spec.task_type,
        "dataset_source": spec.dataset_source,
        "explainer": spec.explainer,
    }
    for name, payload in (
        ("pipeline", pipeline_payload),
        ("dataset", dataset_payload),
        ("split", split_payload),
        ("preprocessor", preprocessor_payload),
        ("model", model_payload),
        ("prediction", prediction_payload),
        ("explanation", explanation_payload),
        ("run", run_payload),
    ):
        _write(output / f"{name}_manifest.json", payload)
    if spec.pipeline_id.startswith("ext3"):
        tracking = output / "mlruns"
        mlflow.set_tracking_uri(tracking.resolve().as_uri())
        mlflow.set_experiment("external-pipeline-fixture")
        with mlflow.start_run(run_name=f"{spec.pipeline_id}-{variant}") as active:
            mlflow.log_params({"alpha": model.alpha, "l1_ratio": model.l1_ratio, "variant": variant})
            mlflow.log_metric("prediction", model_output)
            mlflow.log_artifact(str(artifact_path), artifact_path="model")
            mlflow.set_tags({"model_sha256": model_sha, "registered_model_version": variant})
            run_payload["mlflow_run_id"] = active.info.run_id
        _write(output / "run_manifest.json", run_payload)
    checks = {path.name: _sha(path) for path in sorted(output.glob("*")) if path.is_file()}
    _write(output / "SHA256SUMS.json", checks)
    return output


def prepare_all(root: Path) -> tuple[Path, ...]:
    outputs = []
    for spec in SPECS:
        outputs.append(run_external_pipeline(spec, root, retrained=False))
        outputs.append(run_external_pipeline(spec, root, retrained=True))
    return tuple(outputs)


if __name__ == "__main__":
    prepare_all(Path(__file__).resolve().parent / "fixtures")
