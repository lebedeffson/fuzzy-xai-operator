"""Optional observation of caller-initiated sklearn output calls.

The observer NEVER invokes predict()/predict_proba(), trains, or repairs.
This module records a successful user-requested call's input and returned
output fingerprints, then compares them with a separately supplied report.
It does not attest to the estimator identity, quality, or label truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from .fit_provenance import FeatureSpace, _hashes, _matrix, _target_matrix

OutputKind = Literal["predict", "predict_proba"]
Status = Literal["ok", "problem", "warning", "unknown", "not_applicable"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class OutputCallRecord:
    kind: OutputKind
    feature_space: FeatureSpace
    estimator_type: str
    row_ids: tuple[str, ...]
    n_features: int
    input_row_sha256: tuple[str, ...]
    input_batch_sha256: str
    output_status: Literal["witnessed", "unavailable"]
    output_shape: tuple[int, int] | None
    output_batch_sha256: str | None
    class_order_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OutputTraceCheck:
    code: str
    status: Status
    detail: str
    observed: int | None = None


def _class_order_hash(raw: Any) -> str | None:
    if raw is None:
        return None
    try:
        arr = np.asarray(raw)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1 or len(arr) < 2:
        return None
    classes = tuple(str(value) for value in arr.tolist())
    if len(set(classes)) != len(classes):
        return None
    payload = json.dumps(classes, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b"FXAI-CLASS-ORDER-V1" + payload).hexdigest()


class OutputRecorder:
    """Opt-in transparent wrapper around a *caller-initiated* inference call."""

    def __init__(self) -> None:
        self._records: list[OutputCallRecord] = []

    @property
    def records(self) -> tuple[OutputCallRecord, ...]:
        return tuple(self._records)

    def _invoke(
        self,
        kind: OutputKind,
        estimator: Any,
        X: Any,
        *,
        row_ids: tuple[str, ...],
        feature_space: FeatureSpace,
        kwargs: dict[str, Any],
    ) -> Any:
        if feature_space not in ("raw", "transformed"):
            raise ValueError("feature_space must be raw or transformed")
        matrix = _matrix(X)
        ids = tuple(row_ids)
        if len(ids) != len(matrix) or any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids):
            raise ValueError("Output row IDs must be nonempty, unique, and match the input row count")
        row_hashes, input_hash = _hashes(matrix)
        method = getattr(estimator, kind)
        returned = method(X, **kwargs)
        output_matrix = _target_matrix(returned, len(matrix))
        if output_matrix is None:
            output_status: Literal["witnessed", "unavailable"] = "unavailable"
            output_shape = None
            output_hash = None
        else:
            output_status = "witnessed"
            output_shape = (int(output_matrix.shape[0]), int(output_matrix.shape[1]))
            _, output_hash = _hashes(output_matrix)
        class_hash = _class_order_hash(getattr(estimator, "classes_", None)) if kind == "predict_proba" else None
        self._records.append(
            OutputCallRecord(
                kind=kind,
                feature_space=feature_space,
                estimator_type=type(estimator).__name__,
                row_ids=ids,
                n_features=int(matrix.shape[1]),
                input_row_sha256=row_hashes,
                input_batch_sha256=input_hash,
                output_status=output_status,
                output_shape=output_shape,
                output_batch_sha256=output_hash,
                class_order_sha256=class_hash,
            )
        )
        return returned

    def predict(
        self,
        estimator: Any,
        X: Any,
        *,
        row_ids: tuple[str, ...],
        feature_space: FeatureSpace = "raw",
        **kwargs: Any,
    ) -> Any:
        """Forward one explicit predict call and preserve its exact result."""
        return self._invoke("predict", estimator, X, row_ids=row_ids, feature_space=feature_space, kwargs=kwargs)

    def predict_proba(
        self,
        estimator: Any,
        X: Any,
        *,
        row_ids: tuple[str, ...],
        feature_space: FeatureSpace = "raw",
        **kwargs: Any,
    ) -> Any:
        """Forward one explicit predict_proba call and preserve its result."""
        return self._invoke("predict_proba", estimator, X, row_ids=row_ids, feature_space=feature_space, kwargs=kwargs)


def _parse_records(raw: Any) -> tuple[OutputCallRecord, ...]:
    if not isinstance(raw, (list, tuple)):
        raise TypeError("Output call records must be a sequence")
    parsed: list[OutputCallRecord] = []
    for item in raw:
        try:
            record = item if isinstance(item, OutputCallRecord) else OutputCallRecord(**item)
            if record.kind not in ("predict", "predict_proba") or record.feature_space not in ("raw", "transformed"):
                raise ValueError("Unknown output call kind or feature space")
            if not isinstance(record.estimator_type, str) or not record.estimator_type:
                raise ValueError("Missing estimator type")
            if not isinstance(record.n_features, int) or isinstance(record.n_features, bool) or record.n_features < 1:
                raise ValueError("Invalid feature count")
            if not isinstance(record.row_ids, (tuple, list)) or not record.row_ids:
                raise ValueError("Missing row IDs")
            if len(record.row_ids) != len(record.input_row_sha256) or len(set(record.row_ids)) != len(record.row_ids):
                raise ValueError("Row ID/hash lengths or uniqueness differ")
            if any(not isinstance(v, str) or not v for v in record.row_ids):
                raise ValueError("Row IDs must be nonempty strings")
            if any(not isinstance(v, str) or not _SHA256.fullmatch(v) for v in record.input_row_sha256):
                raise ValueError("Invalid input row digest")
            if not isinstance(record.input_batch_sha256, str) or not _SHA256.fullmatch(record.input_batch_sha256):
                raise ValueError("Invalid input batch digest")
            prefix = (
                b"FXAI-BATCH-V1"
                + len(record.row_ids).to_bytes(8, "big")
                + record.n_features.to_bytes(8, "big")
            )
            recomputed = hashlib.sha256(
                prefix + b"".join(bytes.fromhex(v) for v in record.input_row_sha256)
            ).hexdigest()
            if recomputed != record.input_batch_sha256:
                raise ValueError("Inconsistent input batch fingerprint")
            if record.output_status not in ("witnessed", "unavailable"):
                raise ValueError("Invalid output witness state")
            if record.output_status == "witnessed":
                shape = record.output_shape
                if (
                    not isinstance(shape, (tuple, list))
                    or len(shape) != 2
                    or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in shape)
                    or shape[0] != len(record.row_ids)
                    or not isinstance(record.output_batch_sha256, str)
                    or not _SHA256.fullmatch(record.output_batch_sha256)
                ):
                    raise ValueError("Invalid witnessed output shape or hash")
            elif record.output_shape is not None or record.output_batch_sha256 is not None:
                raise ValueError("Unavailable output cannot have an output hash or shape")
            if record.class_order_sha256 is not None and (
                record.kind != "predict_proba"
                or not isinstance(record.class_order_sha256, str)
                or not _SHA256.fullmatch(record.class_order_sha256)
            ):
                raise ValueError("Invalid class-order commitment")
            parsed.append(record)
        except (TypeError, ValueError, AttributeError, OverflowError) as exc:
            raise ValueError("Invalid output-call record") from exc
    return tuple(parsed)


def audit_output_calls(
    raw: Any,
    *,
    evaluation_features: Any | None = None,
    evaluation_ids: tuple[str, ...] | None = None,
    predictions: Any | None = None,
    probabilities: Any | None = None,
    probability_classes: tuple[str, ...] | None = None,
    task: str | None = None,
) -> tuple[OutputTraceCheck, ...]:
    """Compare recorded calls with a report without performing inference.

    Exact full-batch matching is intentional; partial or multiple calls
    require explicit aggregation outside this narrow adapter.
    """
    if raw is None:
        return (OutputTraceCheck("output_trace", "unknown", "No witnessed prediction or probability calls"),)
    try:
        records = _parse_records(raw)
    except (TypeError, ValueError):
        return (OutputTraceCheck("output_trace", "problem", "Malformed output-call records"),)

    findings: list[OutputTraceCheck] = []
    expected_inputs: np.ndarray | None
    try:
        expected_inputs = _matrix(evaluation_features) if evaluation_features is not None else None
    except ValueError:
        expected_inputs = None

    for kind, supplied in (("predict", predictions), ("predict_proba", probabilities)):
        prefix = f"output_trace_{kind}"
        matching = [record for record in records if record.kind == kind]
        if kind == "predict_proba" and task in ("regression", "clustering"):
            if not matching and supplied is None:
                findings.append(OutputTraceCheck(prefix, "not_applicable", "Classification probabilities are not required for this task"))
            else:
                findings.append(OutputTraceCheck(prefix, "warning", "Probability evidence is outside the declared task adapter"))
            continue
        if not matching:
            findings.append(OutputTraceCheck(prefix, "unknown", "No matching recorded model-output call"))
            continue
        if len(matching) != 1:
            findings.append(OutputTraceCheck(prefix, "unknown", "Several output calls exist without a declared batch-selection rule", len(matching)))
            continue
        record = matching[0]
        if evaluation_ids is None:
            findings.append(OutputTraceCheck(prefix + "_row_ids", "unknown", "Evaluation source row IDs are absent"))
        elif tuple(record.row_ids) != tuple(evaluation_ids):
            findings.append(OutputTraceCheck(prefix + "_row_ids", "problem", "Recorded output row order differs from declared evaluation IDs"))
        else:
            findings.append(OutputTraceCheck(prefix + "_row_ids", "ok", "Recorded evaluation row order matches declared IDs"))

        if record.feature_space != "raw":
            findings.append(OutputTraceCheck(prefix + "_inputs", "unknown", "Transformed inputs need an explicitly recorded transformation chain"))
        elif expected_inputs is None:
            findings.append(OutputTraceCheck(prefix + "_inputs", "unknown", "Evaluation feature matrix cannot be compared"))
        else:
            _, expected_input_hash = _hashes(expected_inputs)
            if record.input_batch_sha256 != expected_input_hash:
                findings.append(OutputTraceCheck(prefix + "_inputs", "problem", "The witnessed inference input batch differs from report features"))
            else:
                findings.append(OutputTraceCheck(prefix + "_inputs", "ok", "Recorded numeric input values and order match report"))

        if supplied is None or record.output_status != "witnessed":
            findings.append(OutputTraceCheck(prefix + "_values", "unknown", "Reported numeric output or witnessed value fingerprint is unavailable"))
        else:
            values = _target_matrix(supplied, len(record.row_ids))
            if values is None:
                findings.append(OutputTraceCheck(prefix + "_values", "unknown", "Numeric output cannot be compared without silent type coercion"))
            else:
                _, actual_hash = _hashes(values)
                mismatch = (
                    tuple(values.shape) != tuple(record.output_shape or ())
                    or actual_hash != record.output_batch_sha256
                )
                findings.append(
                    OutputTraceCheck(
                        prefix + "_values",
                        "problem" if mismatch else "ok",
                        "Reported values differ from witnessed inference result" if mismatch else "Reported output values match the recorded call",
                    )
                )

        if kind == "predict_proba":
            claimed = _class_order_hash(probability_classes)
            if claimed is None or record.class_order_sha256 is None:
                findings.append(OutputTraceCheck(prefix + "_classes", "unknown", "No comparable class-column order evidence"))
            elif claimed != record.class_order_sha256:
                findings.append(OutputTraceCheck(prefix + "_classes", "problem", "Reported probability-column classes differ from witnessed model class order"))
            else:
                findings.append(OutputTraceCheck(prefix + "_classes", "ok", "Declared probability classes match recorded textual order"))
    return tuple(findings)
