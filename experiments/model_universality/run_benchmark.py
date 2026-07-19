from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sklearn
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge, RidgeClassifier, SGDClassifier, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from fuzzyxai import FuzzyXAI, run_adapter_conformance


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "release_evidence/model_universality"
SEED = 42


class GenericProbaModel:
    def __init__(self, fitted: Any):
        self.fitted = fitted
        self.classes_ = fitted.classes_

    def predict(self, values: Any) -> Any:
        return self.fitted.predict(values)

    def predict_proba(self, values: Any) -> Any:
        return self.fitted.predict_proba(values)


class NativeRuleBenchmarkModel(GenericProbaModel):
    def __init__(self, fitted: Any):
        super().__init__(fitted)
        self.rules_ = [
            {
                "rule_id": "R-native-1",
                "antecedents": ["feature_0 is high"],
                "consequent": "1",
                "activation": 0.74,
                "coverage": 0.31,
                "precision": 0.82,
                "human_text": "feature_0 is high -> class 1",
            }
        ]


class CallableBenchmarkModel:
    def __init__(self, fitted: Any):
        self.fitted = fitted

    def __call__(self, values: Any) -> Any:
        return self.fitted.predict(values)


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def dataset_card(name: str, values: np.ndarray, labels: np.ndarray, task: str) -> dict[str, Any]:
    payload = np.ascontiguousarray(values).tobytes() + np.ascontiguousarray(labels).tobytes()
    return {
        "dataset_id": name,
        "source": f"sklearn.datasets.make_{'regression' if task == 'regression' else 'classification'}",
        "license": "synthetic generated data; no external records",
        "task": task,
        "objects": int(values.shape[0]),
        "features": int(values.shape[1]),
        "split_seed": SEED,
        "data_sha256": sha256_bytes(payload),
        "preprocessing": "none; pipeline cases declare StandardScaler separately",
    }


def classification_models() -> list[tuple[str, Callable[[], Any], str]]:
    return [
        ("logistic_binary", lambda: LogisticRegression(max_iter=400, random_state=SEED), "binary"),
        ("sgd_binary", lambda: SGDClassifier(loss="log_loss", max_iter=500, random_state=SEED), "binary"),
        ("ridge_binary", RidgeClassifier, "binary"),
        ("linear_svc_binary", lambda: LinearSVC(random_state=SEED), "binary"),
        ("rbf_svc_binary", lambda: SVC(kernel="rbf", probability=True, random_state=SEED), "binary"),
        ("linear_svc_kernel_binary", lambda: SVC(kernel="linear", probability=True, random_state=SEED), "binary"),
        ("tree_gini_binary", lambda: DecisionTreeClassifier(max_depth=4, random_state=SEED), "binary"),
        ("tree_entropy_binary", lambda: DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=SEED), "binary"),
        ("random_forest_binary", lambda: RandomForestClassifier(n_estimators=12, max_depth=4, random_state=SEED), "binary"),
        ("extra_trees_binary", lambda: ExtraTreesClassifier(n_estimators=12, max_depth=4, random_state=SEED), "binary"),
        ("gradient_boosting_binary", lambda: GradientBoostingClassifier(n_estimators=12, max_depth=2, random_state=SEED), "binary"),
        ("hist_gradient_boosting_binary", lambda: HistGradientBoostingClassifier(max_iter=20, random_state=SEED), "binary"),
        ("adaboost_binary", lambda: AdaBoostClassifier(n_estimators=12, random_state=SEED), "binary"),
        ("knn_binary", lambda: KNeighborsClassifier(n_neighbors=5), "binary"),
        ("gaussian_nb_binary", GaussianNB, "binary"),
        ("pipeline_logistic_binary", lambda: Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=400, random_state=SEED))]), "binary"),
        ("logistic_multiclass", lambda: LogisticRegression(max_iter=400, random_state=SEED), "multiclass"),
        ("tree_multiclass", lambda: DecisionTreeClassifier(max_depth=4, random_state=SEED), "multiclass"),
        ("random_forest_multiclass", lambda: RandomForestClassifier(n_estimators=12, max_depth=4, random_state=SEED), "multiclass"),
        ("knn_multiclass", lambda: KNeighborsClassifier(n_neighbors=5), "multiclass"),
        ("gaussian_nb_multiclass", GaussianNB, "multiclass"),
    ]


def regression_models() -> list[tuple[str, Callable[[], Any]]]:
    return [
        ("linear_regression", LinearRegression),
        ("ridge_regression", Ridge),
        ("lasso_regression", lambda: Lasso(alpha=0.01, max_iter=1000, random_state=SEED)),
        ("elastic_net_regression", lambda: ElasticNet(alpha=0.01, max_iter=1000, random_state=SEED)),
        ("sgd_regression", lambda: SGDRegressor(max_iter=1000, random_state=SEED)),
        ("tree_regression", lambda: DecisionTreeRegressor(max_depth=4, random_state=SEED)),
        ("random_forest_regression", lambda: RandomForestRegressor(n_estimators=12, max_depth=4, random_state=SEED)),
        ("gradient_boosting_regression", lambda: GradientBoostingRegressor(n_estimators=12, max_depth=2, random_state=SEED)),
        ("knn_regression", lambda: KNeighborsRegressor(n_neighbors=5)),
        ("svr_regression", lambda: SVR(kernel="rbf")),
    ]


def _run_configuration(
    config_id: str,
    model: Any,
    task: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    output: Path,
) -> dict[str, Any]:
    feature_names = [f"feature_{index}" for index in range(train_x.shape[1])]
    fx = FuzzyXAI.wrap(model, task=task)
    result = fx.explain_one(
        test_x[0],
        object_id=f"{config_id}_object",
        feature_names=feature_names,
        reference_data=train_x[:40],
        reference_labels=train_y[:40].tolist(),
    )
    conformance = run_adapter_conformance(fx.model_adapter, sample_batch=test_x[:2])
    graph_errors = result.explanation_graph.validate_reachability()
    quality = result.quality_report()
    payload = {
        "config_id": config_id,
        "task": task,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "adapter_id": fx.model_adapter.adapter_id,
        "model_family": str(getattr(fx.model_adapter, "model_family", "legacy")),
        "prediction_parity": conformance.status == "pass" and not any(item.check_id == "prediction_parity" and item.status == "fail" for item in conformance.checks),
        "conformance": conformance.to_dict(),
        "capabilities": fx.capability_report(),
        "graph_errors": list(graph_errors),
        "human_explanation": result.explain_for().to_dict(include_technical_trace=False),
        "quality": quality.to_dict(),
        "missing_channels": list(result.missing_channels),
        "status": "pass" if conformance.status == "pass" and not graph_errors else "fail",
    }
    write_json(output / "conformance_reports" / f"{config_id}.json", payload)
    write_json(output / "human_explanations" / f"{config_id}.json", payload["human_explanation"])
    write_json(output / "quality_reports" / f"{config_id}.json", payload["quality"])
    return payload


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    binary_x, binary_y = make_classification(n_samples=180, n_features=8, n_informative=6, n_redundant=0, weights=[0.7, 0.3], random_state=SEED)
    multi_x, multi_y = make_classification(n_samples=180, n_features=8, n_informative=6, n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=SEED)
    regression_x, regression_y = make_regression(n_samples=180, n_features=8, n_informative=6, noise=0.2, random_state=SEED)
    cards = {
        "binary": dataset_card("synthetic_binary_imbalanced_seed42", binary_x, binary_y, "binary_classification"),
        "multiclass": dataset_card("synthetic_multiclass_seed42", multi_x, multi_y, "multiclass_classification"),
        "regression": dataset_card("synthetic_regression_seed42", regression_x, regression_y, "regression"),
    }
    write_json(output / "dataset_cards.json", cards)
    rows: list[dict[str, Any]] = []
    for config_id, factory, dataset in classification_models():
        values, labels = (binary_x, binary_y) if dataset == "binary" else (multi_x, multi_y)
        model = factory().fit(values[:140], labels[:140])
        task = "binary_classification" if dataset == "binary" else "multiclass_classification"
        rows.append(_run_configuration(config_id, model, task, values[:140], labels[:140], values[140:], output))

    fitted_logistic = LogisticRegression(max_iter=400, random_state=SEED).fit(binary_x[:140], binary_y[:140])
    generic_models = [
        ("generic_predict_proba", GenericProbaModel(fitted_logistic)),
        ("native_rule_model", NativeRuleBenchmarkModel(fitted_logistic)),
        ("callable_black_box", CallableBenchmarkModel(fitted_logistic)),
    ]
    for config_id, model in generic_models:
        rows.append(_run_configuration(config_id, model, "binary_classification", binary_x[:140], binary_y[:140], binary_x[140:], output))

    for config_id, factory in regression_models():
        model = factory().fit(regression_x[:140], regression_y[:140])
        rows.append(_run_configuration(config_id, model, "regression", regression_x[:140], regression_y[:140], regression_x[140:], output))

    optional = []
    for package, adapters in {
        "xgboost": ("xgboost_classifier", "xgboost_regressor"),
        "lightgbm": ("lightgbm_classifier", "lightgbm_regressor"),
        "catboost": ("catboost_classifier", "catboost_regressor"),
        "torch": ("torch_classifier", "torch_regressor"),
        "tensorflow": ("keras_classifier", "keras_regressor"),
        "onnxruntime": ("onnx_classifier", "onnx_regressor"),
    }.items():
        try:
            module = __import__(package)
            status = "implemented_not_executed"
            version = getattr(module, "__version__", "unknown")
        except ImportError:
            status = "dependency_unavailable"
            version = None
        optional.extend({"config_id": item, "package": package, "status": status, "version": version} for item in adapters)
    write_json(output / "optional_integrations.json", optional)

    support_fields = ["config_id", "task", "model_class", "adapter_id", "model_family", "prediction_parity", "status"]
    with (output / "support_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=support_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in support_fields} for row in rows)
    summary = {
        "schema_version": "1.0",
        "framework_version": "1.3.0rc1",
        "sklearn_version": sklearn.__version__,
        "seed": SEED,
        "classification_configurations": sum(row["task"] != "regression" for row in rows),
        "regression_configurations": sum(row["task"] == "regression" for row in rows),
        "verified_configurations": len(rows),
        "prediction_parity_rate": float(np.mean([row["prediction_parity"] for row in rows])),
        "graph_validation_rate": float(np.mean([not row["graph_errors"] for row in rows])),
        "conformance_rate": float(np.mean([row["status"] == "pass" for row in rows])),
        "optional_integrations": optional,
        "release_claim": "Only configurations with status=pass are verified; unavailable or unexecuted optional integrations are not claimed.",
    }
    write_json(output / "summary.json", summary)
    manifest_entries = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name not in {"manifest.json", "checksums.sha256"}):
        manifest_entries.append({"path": path.relative_to(output).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    manifest = {"schema_version": "1.0", "summary": summary, "files": manifest_entries}
    write_json(output / "manifest.json", manifest)
    checksums = [f"{item['sha256']}  {item['path']}" for item in manifest_entries]
    checksums.append(f"{hashlib.sha256((output / 'manifest.json').read_bytes()).hexdigest()}  manifest.json")
    (output / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="ascii")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
