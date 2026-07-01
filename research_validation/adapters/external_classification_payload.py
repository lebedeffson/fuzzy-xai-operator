from __future__ import annotations


def make_feature_importance(total: float, prefix: str = "x") -> dict[str, float]:
    ratios = [0.32, 0.24, 0.18, 0.14, 0.12]
    return {f"{prefix}{index}": round(total * ratio, 6) for index, ratio in enumerate(ratios, start=1)}


def make_feature_values(prefix: str = "x") -> dict[str, float]:
    return {f"{prefix}{index}": round(0.1 * index, 3) for index in range(1, 6)}


def make_payload(experiment: dict) -> dict:
    feature_prefix = "pixel" if "image" in experiment["task_type"] else "f"
    context = {
        "experiment_id": experiment["id"],
        "task_type": experiment["task_type"],
        "perturbation": experiment["perturbation"],
        "representation_class": experiment.get("representation_class", "F0"),
        "prediction_interval_width": float(experiment.get("prediction_interval_width", 0.0)),
        "conflict_component": float(experiment.get("conflict_component", 0.0)),
        "noise_ratio": float(experiment.get("noise_ratio", 0.0)),
        "occlusion_rate": float(experiment.get("occlusion_rate", 0.0)),
    }
    return {
        "scenario_id": "external_wine_classification",
        "source_type": experiment.get("source_type", "tabular"),
        "model_name": experiment["model"],
        "dataset_name": experiment["dataset"],
        "predicted_class": 1,
        "class_probability": float(experiment["probability"]),
        "feature_values": make_feature_values(feature_prefix),
        "feature_importance": make_feature_importance(float(experiment["importance_sum"]), feature_prefix),
        "top_k_importance": 5,
        "quality_metrics": {
            "missing_rate": float(experiment.get("missing_rate", 0.0)),
            "feature_range_violation": float(experiment.get("range_violation", 0.0)),
        },
        "context_values": context,
    }
