"""Opt-in, source-bound fit-call witnesses for the read-only ML observer.

Training is NEVER triggered by the observer. Users may explicitly route
their own existing estimator.fit call through FitRecorder.fit(...), which
forwards arguments unchanged and records the successful invocation.
Metadata recorded this way is *locally observed*, not remotely attested;
unwrapped calls are invisible and callers can still lie about row IDs.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

FitRole = Literal["model", "preprocessor"]
FeatureSpace = Literal["raw", "transformed"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FitCallRecord:
    role: FitRole
    feature_space: FeatureSpace
    estimator_type: str
    row_ids: tuple[str, ...]
    n_features: int
    row_sha256: tuple[str, ...]
    batch_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FitTraceCheck:
    code: str
    status: Literal["ok", "problem", "warning", "unknown", "not_applicable"]
    detail: str
    observed: int | None = None


def _matrix(value: Any) -> np.ndarray:
    """Numeric finite snapshot, in a fixed floating-point representation."""
    try:
        m = np.asarray(value, dtype="<f8")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("fit witness requires finite numeric tabular inputs") from exc
    if m.ndim != 2 or 0 in m.shape or not np.isfinite(m).all():
        raise ValueError("fit witness requires a nonempty finite 2-D input")
    snapshot = np.array(m, dtype="<f8", order="C", copy=True)
    snapshot[snapshot == 0.0] = 0.0  # Canonicalize -0.0.
    return snapshot


def _hashes(m: np.ndarray) -> tuple[tuple[str, ...], str]:
    width = int(m.shape[1])
    row_sha256 = tuple(
        hashlib.sha256(b"FXAI-ROW-V1" + width.to_bytes(8, "big") + row.tobytes()).hexdigest()
        for row in m
    )
    payload = b"FXAI-BATCH-V1" + len(m).to_bytes(8, "big") + width.to_bytes(8, "big")
    batch = hashlib.sha256(payload + b"".join(bytes.fromhex(h) for h in row_sha256)).hexdigest()
    return row_sha256, batch


class FitRecorder:
    """A transparent explicit fit-call wrapper; no global hooks/monkeypatching."""

    def __init__(self) -> None:
        self._records: list[FitCallRecord] = []

    @property
    def records(self) -> tuple[FitCallRecord, ...]:
        return tuple(self._records)

    def fit(
        self,
        estimator: Any,
        X: Any,
        y: Any | None = None,
        *,
        role: FitRole,
        row_ids: tuple[str, ...],
        feature_space: FeatureSpace = "raw",
        **fit_kwargs: Any,
    ) -> Any:
        """Call estimator.fit once at caller request and witness source inputs.

        No record is appended when .fit raises. Snapshotting precedes .fit
        because some estimators may mutate input buffers in-place.
        """

        if role not in ("model", "preprocessor"):
            raise ValueError("fit role must be model or preprocessor")
        if feature_space not in ("raw", "transformed"):
            raise ValueError("feature_space must be raw or transformed")
        m = _matrix(X)
        ids = tuple(row_ids)
        if len(ids) != len(m) or any(not isinstance(v, str) or not v for v in ids):
            raise ValueError("row_ids must contain one nonempty source ID per fitted row")
        hashed, batch = _hashes(m)
        if y is None:
            result = estimator.fit(X, **fit_kwargs)
        else:
            result = estimator.fit(X, y, **fit_kwargs)
        self._records.append(
            FitCallRecord(
                role=role,
                feature_space=feature_space,
                estimator_type=type(estimator).__name__,
                row_ids=ids,
                n_features=int(m.shape[1]),
                row_sha256=hashed,
                batch_sha256=batch,
            )
        )
        return result


def _parse_records(raw: Any) -> tuple[FitCallRecord, ...]:
    if not isinstance(raw, (tuple, list)):
        raise TypeError("fit trace must be an array of records")
    output = []
    for item in raw:
        try:
            record = item if isinstance(item, FitCallRecord) else FitCallRecord(**item)
            if record.role not in ("model", "preprocessor") or record.feature_space not in ("raw", "transformed"):
                raise ValueError("unsupported fit role or feature space")
            if not isinstance(record.estimator_type, str) or not record.estimator_type:
                raise ValueError("missing estimator type")
            if not isinstance(record.n_features, int) or isinstance(record.n_features, bool) or record.n_features < 1:
                raise ValueError("invalid feature width")
            if not isinstance(record.row_ids, (list, tuple)) or not record.row_ids:
                raise ValueError("empty fit row identities")
            if len(record.row_ids) != len(record.row_sha256):
                raise ValueError("fit IDs and row digests have different lengths")
            if any(not isinstance(v, str) or not v for v in record.row_ids):
                raise ValueError("invalid fit source row IDs")
            if any(not isinstance(v, str) or not _SHA256.fullmatch(v) for v in record.row_sha256):
                raise ValueError("invalid row digest")
            if not isinstance(record.batch_sha256, str) or not _SHA256.fullmatch(record.batch_sha256):
                raise ValueError("invalid batch digest")
            payload = b"FXAI-BATCH-V1" + len(record.row_ids).to_bytes(8, "big") + record.n_features.to_bytes(8, "big")
            calculated = hashlib.sha256(payload + b"".join(bytes.fromhex(v) for v in record.row_sha256)).hexdigest()
            if calculated != record.batch_sha256:
                raise ValueError("fit batch digest is inconsistent with row digest claims")
        except (TypeError, KeyError, ValueError, AttributeError, OverflowError) as exc:
            raise ValueError("invalid or internally inconsistent fit trace record") from exc
        output.append(record)
    return tuple(output)


def audit_fit_trace(
    raw: Any,
    *,
    train_features: Any | None = None,
    evaluation_features: Any | None = None,
    train_ids: tuple[str, ...] | None = None,
    evaluation_ids: tuple[str, ...] | None = None,
    model_fit_ids: tuple[str, ...] | None = None,
    preprocessor_fit_ids: tuple[str, ...] | None = None,
    preprocessor_applied: bool | None = None,
) -> tuple[FitTraceCheck, ...]:
    """Evaluate optional fit witnesses without invoking or modifying a model.

    A non-overlapping trace *only* describes recorded calls. Other unwrapped
    fitting, dishonest source IDs, or altered code paths are not excluded.
    Raw feature row hashes can be compared to supplied train/test arrays;
    transformed inputs require a separately instrumented transform chain.
    """

    if raw is None:
        return (
            FitTraceCheck("fit_trace", "unknown", "No instrumented fit events supplied"),
        )
    try:
        records = _parse_records(raw)
    except (TypeError, ValueError):
        return (FitTraceCheck("fit_trace", "problem", "Malformed or internally inconsistent fit records"),)

    checks: list[FitTraceCheck] = []
    identifiers = set(evaluation_ids) if evaluation_ids is not None else None
    reference: dict[str, str] = {}
    comparable = True
    for declared, values in (
        (train_ids, train_features),
        (evaluation_ids, evaluation_features),
    ):
        if declared is None or values is None:
            comparable = False
            continue
        try:
            m = _matrix(values)
        except ValueError:
            comparable = False
            continue
        if len(declared) != len(m) or len(set(declared)) != len(declared):
            comparable = False
            continue
        hashes, _ = _hashes(m)
        for row_id, row_hash in zip(declared, hashes):
            if row_id in reference and reference[row_id] != row_hash:
                comparable = False
            reference[row_id] = row_hash

    for role in ("model", "preprocessor"):
        records_for_role = tuple(r for r in records if r.role == role)
        code_prefix = f"fit_trace_{role}"
        if role == "preprocessor" and preprocessor_applied is False and not records_for_role:
            checks.append(FitTraceCheck(f"{code_prefix}_isolation", "not_applicable", "No fitted preprocessor declared"))
            checks.append(FitTraceCheck(f"{code_prefix}_content", "not_applicable", "No fitted preprocessor declared"))
            continue
        if not records_for_role:
            checks.append(FitTraceCheck(f"{code_prefix}_isolation", "unknown", "No successful fit call was observed"))
            checks.append(FitTraceCheck(f"{code_prefix}_content", "unknown", "No fitted inputs are available"))
            continue
        if role == "preprocessor" and preprocessor_applied is False:
            checks.append(FitTraceCheck(f"{code_prefix}_isolation", "problem", "Recorded preprocessing contradicts declared absence"))
        elif identifiers is None:
            checks.append(FitTraceCheck(f"{code_prefix}_isolation", "unknown", "Cannot check recorded fit rows without evaluation IDs"))
        else:
            overlap = sum(identifier in identifiers for record in records_for_role for identifier in record.row_ids)
            checks.append(
                FitTraceCheck(
                    f"{code_prefix}_isolation",
                    "problem" if overlap else "ok",
                    "A witnessed fit received evaluation row IDs" if overlap else "No evaluation row IDs in witnessed fit calls",
                    overlap,
                )
            )

        declared_ids = model_fit_ids if role == "model" else preprocessor_fit_ids
        if declared_ids is not None:
            witnessed_ids = {v for record in records_for_role for v in record.row_ids}
            checks.append(
                FitTraceCheck(
                    f"{code_prefix}_declarations",
                    "warning" if witnessed_ids != set(declared_ids) else "ok",
                    "Declared fit IDs disagree with witnessed fit calls" if witnessed_ids != set(declared_ids) else "Declared IDs agree with witnessed calls",
                )
            )
        raw_records = [record for record in records_for_role if record.feature_space == "raw"]
        if not raw_records or not comparable:
            checks.append(FitTraceCheck(f"{code_prefix}_content", "unknown", "Raw fit content cannot be compared to complete declared source matrices"))
        else:
            unmatched = sum(
                reference.get(row_id) != row_hash
                for record in raw_records
                for row_id, row_hash in zip(record.row_ids, record.row_sha256)
            )
            checks.append(
                FitTraceCheck(
                    f"{code_prefix}_content",
                    "problem" if unmatched else "ok",
                    "Raw fitted rows contradict source content declarations" if unmatched else "Raw fitted rows match declared source values by row ID",
                    unmatched,
                )
            )
    return tuple(checks)
