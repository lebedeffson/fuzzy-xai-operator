from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fuzzyxai import FuzzyXAI
from fuzzyxai.adapters import get_adapter
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def run_example(output: Path) -> dict[str, object]:
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tracking = output / "mlruns"
    mlflow.set_tracking_uri(tracking.as_uri())
    mlflow.set_experiment("fuzzyxai-local-integration")

    iris = load_iris()
    mask = iris.target < 2
    features = iris.data[mask]
    labels = iris.target[mask]
    model = LogisticRegression(
        max_iter=500,
        random_state=7421,
    ).fit(features, labels)
    model_name = "fuzzyxai-demo"
    with mlflow.start_run() as active:
        mlflow.log_params(
            {
                "model_family": "LogisticRegression",
                "random_state": 7421,
                "dataset": "sklearn_iris_binary",
            }
        )
        mlflow.set_tags(
            {
                "fuzzyxai.integration": "local-demo",
                "fuzzyxai.explainer": "linear-coefficient",
            }
        )
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
            input_example=features[:2],
        )
        run_id = active.info.run_id

    client = MlflowClient()
    versions = client.search_model_versions(
        f"name='{model_name}'"
    )
    version = max(versions, key=lambda item: int(item.version))
    model_uri = f"models:/{model_name}/{version.version}"
    loaded = mlflow.sklearn.load_model(model_uri)
    run = client.get_run(version.run_id)
    sample = features[[0]]
    probability = float(loaded.predict_proba(sample)[0, 1])
    prediction = int(probability >= 0.5)
    raw_contributions = np.abs(
        loaded.coef_[0] * sample[0]
    )
    total = float(raw_contributions.sum()) or 1.0
    contributions = {
        name: float(value / total)
        for name, value in zip(
            iris.feature_names,
            raw_contributions,
            strict=True,
        )
    }
    payload = {
        "scenario_id": "mlflow_tabular_classification",
        "model_name": model_name,
        "dataset_name": "sklearn_iris_binary",
        "predicted_class": prediction,
        "class_probability": probability,
        "feature_values": {
            name: float(value)
            for name, value in zip(
                iris.feature_names,
                sample[0],
                strict=True,
            )
        },
        "feature_importance": contributions,
        "quality_metrics": {
            "missing_rate": 0.0,
            "feature_range_violation": 0.0,
        },
        "run_id": version.run_id,
        "model_version": version.version,
        "artifact_uri": version.source,
        "mlflow_version": mlflow.__version__,
        "mlflow_params": dict(run.data.params),
        "mlflow_tags": dict(run.data.tags),
    }
    route = FuzzyXAI().run_payload(
        payload,
        get_adapter("mlflow_tabular"),
    )
    verification = FuzzyXAI().verify(route)
    route.write_json(output / "mlflow_fuzzyxai_route.json")
    result = {
        "status": (
            "MLFLOW_INTEGRATION_PASS"
            if verification.valid
            else "MLFLOW_INTEGRATION_FAIL"
        ),
        "mlflow_version": mlflow.__version__,
        "scikit_learn_model": "LogisticRegression",
        "tracking_uri": tracking.as_uri(),
        "registered_model_name": model_name,
        "model_version": version.version,
        "run_id": run_id,
        "artifact_uri": version.source,
        "route_id": route.route_id,
        "route_verification": verification.valid,
        "external_network_required": False,
        "scientific_hypothesis": False,
        "product_comparison": False,
    }
    (output / "MLFLOW_INTEGRATION_STATUS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    result = run_example(
        Path("results/integrations/mlflow_demo")
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
