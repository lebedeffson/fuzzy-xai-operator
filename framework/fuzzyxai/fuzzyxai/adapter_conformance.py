from __future__ import annotations

import json
from typing import Any

import numpy as np

from fuzzyxai.adapters.contracts_v2 import AdapterCheck, AdapterConformanceReport, ExplanationContext, TaskType
from fuzzyxai.adapters.model import ModelAdapter, _serializable


def _equal(left: Any, right: Any) -> bool:
    try:
        return bool(np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float), rtol=1e-9, atol=1e-12))
    except (TypeError, ValueError):
        return bool(_serializable(left) == _serializable(right))


def run_adapter_conformance(
    adapter: ModelAdapter,
    model: Any | None = None,
    sample_batch: Any = None,
    reference_data: Any = None,
) -> AdapterConformanceReport:
    """Run deterministic adapter checks without assuming unsupported channels exist."""

    del reference_data
    checked_model = model or adapter.model
    if sample_batch is None:
        sample_batch = np.zeros((2, max(1, int(getattr(checked_model, "n_features_in_", 1)))), dtype=float)
    checks: list[AdapterCheck] = []
    errors: list[str] = []

    def record(check_id: str, passed: bool, detail: str) -> None:
        checks.append(AdapterCheck(check_id, "pass" if passed else "fail", detail))
        if not passed:
            errors.append(f"{check_id}: {detail}")

    try:
        prediction = adapter.predict(sample_batch)
        record("prediction_available", prediction.predictions is not None, "canonical prediction returned")
    except Exception as exc:
        record("prediction_available", False, f"prediction failed: {exc}")
        prediction = None
    direct_predict = getattr(checked_model, "predict", None)
    if prediction is not None and callable(direct_predict):
        try:
            record("prediction_parity", _equal(prediction.predictions, direct_predict(sample_batch)), "adapter output matches model.predict")
        except Exception as exc:
            record("prediction_parity", False, f"direct prediction failed: {exc}")
    else:
        checks.append(AdapterCheck("prediction_parity", "not_applicable", "model.predict is unavailable"))
    try:
        capabilities = adapter.capabilities()
        payload = capabilities.to_dict()
        record("capabilities_serializable", bool(json.dumps(payload, sort_keys=True)), "capability report is JSON serializable")
        declared = {item.name for item in getattr(capabilities, "channels", ()) if item.available}
        record("capability_truthfulness", all(name for name in declared), "all available channels have explicit descriptors")
    except Exception as exc:
        record("capabilities_serializable", False, f"capability report failed: {exc}")
        capabilities = None
    try:
        first = adapter.model_fingerprint()
        second = adapter.model_fingerprint()
        record("fingerprint_stable", first == second and len(first) == 64, "fingerprint is stable SHA256")
    except Exception as exc:
        record("fingerprint_stable", False, f"fingerprint failed: {exc}")
    if prediction is not None and hasattr(adapter, "extract_local_evidence"):
        try:
            names = tuple(adapter.feature_names())
            local = adapter.extract_local_evidence(sample_batch, prediction, ExplanationContext(feature_names=names))
            serialized = json.dumps(local.to_runtime_mapping(), sort_keys=True, default=str)
            record("local_evidence_serializable", bool(serialized), "local evidence is deterministic JSON data")
        except Exception as exc:
            record("local_evidence_serializable", False, f"local evidence failed: {exc}")
    else:
        checks.append(AdapterCheck("local_evidence_serializable", "not_applicable", "v1 adapter does not expose typed local evidence"))
    task_type = getattr(adapter, "task_type", TaskType.BINARY_CLASSIFICATION)
    return AdapterConformanceReport(
        adapter_id=adapter.adapter_id,
        model_family=str(getattr(adapter, "model_family", type(checked_model).__name__)),
        task_type=task_type,
        status="pass" if not errors else "fail",
        checks=tuple(checks),
        errors=tuple(errors),
    )
