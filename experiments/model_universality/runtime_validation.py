from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from fuzzyxai import FuzzyXAI, run_adapter_conformance


SEED = 42
SUPPORTED_LIBRARIES = ("sklearn", "xgboost", "lightgbm", "catboost", "torch", "tensorflow", "onnx")


@dataclass(frozen=True)
class RuntimeCase:
    model: Any
    values: np.ndarray
    labels: np.ndarray
    model_class: str
    task_type: str
    library_package: str
    direct_predict: Callable[[np.ndarray], Any]


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    unsigned = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    return json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def report_checksum(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def write_report(path: Path, payload: dict[str, Any]) -> None:
    payload["artifact_sha256"] = report_checksum(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _classification_data(features: int = 4) -> tuple[np.ndarray, np.ndarray]:
    values, labels = make_classification(
        n_samples=64,
        n_features=features,
        n_informative=max(2, features - 1),
        n_redundant=0,
        random_state=SEED,
    )
    return values.astype(np.float32), labels.astype(int)


def _build_sklearn() -> RuntimeCase:
    values, labels = _classification_data()
    model = LogisticRegression(max_iter=200, random_state=SEED).fit(values, labels)
    return RuntimeCase(model, values, labels, type(model).__name__, "binary_classification", "scikit-learn", model.predict)


def _build_xgboost() -> RuntimeCase:
    import xgboost as xgb

    values, labels = _classification_data()
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=SEED, n_jobs=1).fit(values, labels)
    return RuntimeCase(model, values, labels, type(model).__name__, "binary_classification", "xgboost", model.predict)


def _build_lightgbm() -> RuntimeCase:
    import lightgbm as lgb

    values, labels = _classification_data()
    model = lgb.LGBMClassifier(n_estimators=5, max_depth=2, random_state=SEED, n_jobs=1, verbose=-1).fit(values, labels)
    return RuntimeCase(model, values, labels, type(model).__name__, "binary_classification", "lightgbm", model.predict)


def _build_catboost() -> RuntimeCase:
    import catboost

    values, labels = _classification_data()
    model = catboost.CatBoostClassifier(iterations=5, depth=2, random_seed=SEED, thread_count=1, verbose=False).fit(values, labels)
    return RuntimeCase(model, values, labels, type(model).__name__, "binary_classification", "catboost", model.predict)


def _build_torch() -> RuntimeCase:
    import torch

    torch.manual_seed(SEED)
    values, labels = _classification_data(3)
    model = torch.nn.Sequential(torch.nn.Linear(3, 2))

    def direct(rows: np.ndarray) -> Any:
        was_training = bool(model.training)
        model.eval()
        try:
            with torch.no_grad():
                return torch.argmax(model(torch.as_tensor(rows, dtype=torch.float32)), dim=-1).cpu().numpy()
        finally:
            model.train(was_training)

    return RuntimeCase(model, values, labels, type(model).__name__, "binary_classification", "torch", direct)


def _build_tensorflow() -> RuntimeCase:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    values, labels = _classification_data(3)
    model = tf.keras.Sequential([tf.keras.layers.Input((3,)), tf.keras.layers.Dense(2)])

    def direct(rows: np.ndarray) -> Any:
        output = model(tf.convert_to_tensor(rows, dtype=tf.float32), training=False)
        return np.argmax(np.asarray(output), axis=-1)

    return RuntimeCase(model, values, labels, type(model).__name__, "binary_classification", "tensorflow", direct)


def _build_onnx(directory: Path) -> RuntimeCase:
    import onnx

    helper = onnx.helper
    tensor = onnx.TensorProto
    graph = helper.make_graph(
        [helper.make_node("Sigmoid", ["input"], ["probability"])],
        "fuzzyxai-runtime-validation",
        [helper.make_tensor_value_info("input", tensor.FLOAT, [None, 1])],
        [helper.make_tensor_value_info("probability", tensor.FLOAT, [None, 1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = min(model.ir_version, 9)
    path = directory / "runtime_validation.onnx"
    onnx.save(model, path)
    values = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(-1, 1)
    labels = (values[:, 0] >= 0).astype(int)

    def direct(rows: np.ndarray) -> Any:
        import onnxruntime as ort

        probabilities = ort.InferenceSession(str(path)).run(None, {"input": np.asarray(rows, dtype=np.float32)})[0]
        return (np.asarray(probabilities).reshape(-1) >= 0.5).astype(int)

    return RuntimeCase(path, values, labels, "ONNXModel", "binary_classification", "onnxruntime", direct)


BUILDERS: dict[str, Callable[..., RuntimeCase]] = {
    "sklearn": _build_sklearn,
    "xgboost": _build_xgboost,
    "lightgbm": _build_lightgbm,
    "catboost": _build_catboost,
    "torch": _build_torch,
    "tensorflow": _build_tensorflow,
}


def _equal(left: Any, right: Any) -> bool:
    try:
        return bool(np.allclose(np.asarray(left), np.asarray(right), rtol=1e-7, atol=1e-9))
    except (TypeError, ValueError):
        return left == right


def _top_reasons(result: Any) -> tuple[str, ...]:
    contributions = result.model_evidence.get("contributions", {})
    if not isinstance(contributions, dict):
        return ()
    ranked = sorted(contributions.items(), key=lambda item: abs(float(item[1])), reverse=True)
    return tuple(str(name) for name, _ in ranked[:3])


def _overlap(left: tuple[str, ...], right: tuple[str, ...]) -> float | None:
    if not left or not right:
        return None
    union = set(left) | set(right)
    return len(set(left) & set(right)) / len(union) if union else None


def build_runtime_report(library: str) -> dict[str, Any]:
    started = time.perf_counter()
    warnings: list[str] = []
    with tempfile.TemporaryDirectory(prefix="fuzzyxai-runtime-") as temp:
        case = _build_onnx(Path(temp)) if library == "onnx" else BUILDERS[library]()
        names = [f"feature_{index}" for index in range(case.values.shape[1])]
        fx = FuzzyXAI.wrap(case.model, task=case.task_type)
        result = fx.explain_one(
            case.values[:1],
            object_id=f"{library}_runtime_object",
            feature_names=names,
            reference_data=case.values[:32],
            reference_labels=case.labels[:32].tolist(),
        )
        direct = case.direct_predict(case.values[:2])
        adapted = fx.model_adapter.predict(case.values[:2]).predictions
        parity = _equal(adapted, direct)
        conformance = run_adapter_conformance(fx.model_adapter, sample_batch=case.values[:2])
        graph_errors = list(result.explanation_graph.validate_reachability())
        quality = result.quality_report()
        human = result.explain_for()
        forbidden = re.compile(
            r"\b(?:R\d+|S\d+|E[0-5]|gamma|delta|rho|claim_id|defer_to_human)\b",
            re.IGNORECASE,
        )
        human_checks = {
            "decision_present": bool(human.decision.explanation),
            "reason_count_valid": 1 <= len(human.main_reasons) <= 3,
            "concern_count_valid": len(human.concerns) <= 2,
            "reliability_present": bool(human.reliability.explanation),
            "recommended_action_present": bool(human.recommended_action.explanation),
            "comparisons_present": all(bool(reason.comparison_text) for reason in human.main_reasons),
            "all_fragments_grounded": all(fragment.claim_refs and fragment.evidence_refs for fragment in human.fragments),
            "technical_terms_hidden": forbidden.search(human.user_text) is None,
        }

        repeated = fx.explain_one(case.values[:1], feature_names=names)
        perturbed_values = case.values[:1].copy()
        perturbed_values[0, 0] += np.float32(1e-4)
        perturbed = fx.explain_one(perturbed_values, feature_names=names)
        repeat_overlap = _overlap(_top_reasons(result), _top_reasons(repeated))
        perturbation_overlap = _overlap(_top_reasons(result), _top_reasons(perturbed))
        measured_stability = [value for value in (repeat_overlap, perturbation_overlap) if value is not None]
        stability = float(np.median(measured_stability)) if measured_stability else None

        batch = fx.explain_batch(case.values[:2], object_ids=["runtime_0", "runtime_1"], feature_names=names)
        global_result = fx.explain_global(case.values[:8], case.labels[:8], feature_names=names)
        why_not = result.why_not(1 - int(np.asarray(result.prediction.predictions).reshape(-1)[0]))
        comparison = FuzzyXAI.compare_models(
            {"candidate_a": case.model, "candidate_b": case.model},
            item=case.values[:1],
            reference_data=case.values[:8],
            reference_labels=case.labels[:8].tolist(),
            task=case.task_type,
            feature_names=names,
        )
        api_checks = {
            "explain_one": result.prediction.predictions is not None,
            "explain_batch": len(batch.view_model.trace.get("object_ids", ())) == 2,
            "explain_global": global_result.sample_count == 8,
            "why_not": why_not.status in {"supported", "insufficient_evidence"},
            "compare_models": len(comparison.model_results) == 2,
        }

        capabilities = fx.model_adapter.capabilities()
        descriptors = [item.to_dict() for item in capabilities.channels]
        native_channels = [item["name"] for item in descriptors if item["available"] and item["origin"] == "native"]
        surrogate_channels = [item["name"] for item in descriptors if item["available"] and item["origin"] == "surrogate"]
        derived_channels = [item["name"] for item in descriptors if item["available"] and item["origin"] in {"derived", "derived_from_native"}]
        missing_channels = sorted(set(result.missing_channels))
        if stability is None:
            warnings.append("Top-reason stability is not applicable because this adapter disclosed no local contributions.")
        if quality.status != "pass":
            warnings.extend(quality.limitations)

        version = importlib.metadata.version(case.library_package)
        python_compact = f"py{sys.version_info.major}{sys.version_info.minor}"
        status = (
            "pass"
            if parity
            and conformance.status == "pass"
            and not graph_errors
            and all(api_checks.values())
            and not quality.blocked_channels
            and all(human_checks.values())
            else "failed"
        )
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "report_id": f"{library}_{python_compact}",
            "library": library,
            "library_package": case.library_package,
            "library_version": version,
            "python_version": platform.python_version(),
            "operating_system": platform.platform(),
            "environment": f"optional-runtime-{library}" if library != "sklearn" else "core-model-contracts",
            "model_family": str(getattr(fx.model_adapter, "model_family", library)),
            "model_class": case.model_class,
            "task_type": case.task_type,
            "adapter": type(fx.model_adapter).__name__,
            "adapter_id": fx.model_adapter.adapter_id,
            "sample_size": int(case.values.shape[0]),
            "prediction_parity": 1.0 if parity else 0.0,
            "conformance": 1.0 if conformance.status == "pass" else 0.0,
            "conformance_report": conformance.to_dict(),
            "graph_validation": 1.0 if not graph_errors else 0.0,
            "graph_errors": graph_errors,
            "quality_gate": quality.status,
            "quality": {**quality.to_dict(), "measured_top_reason_stability": stability},
            "api_checks": api_checks,
            "human_explanation_checks": human_checks,
            "native_channels": native_channels,
            "derived_channels": derived_channels,
            "surrogate_channels": surrogate_channels,
            "missing_channels": missing_channels,
            "available_evidence_sources": descriptors,
            "duration_seconds": round(time.perf_counter() - started, 6),
            "warnings": warnings,
            "status": status,
        }
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a runtime-specific FuzzyXAI adapter validation report.")
    parser.add_argument("--library", required=True, choices=SUPPORTED_LIBRARIES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_runtime_report(args.library)
    python_compact = f"py{sys.version_info.major}{sys.version_info.minor}"
    destination = args.output / f"adapter_report_{args.library}_{python_compact}.json"
    write_report(destination, payload)
    print(f"PASS: runtime_report_{args.library} {destination}")


if __name__ == "__main__":
    main()
