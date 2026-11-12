"""Single-step numeric transform witnesses: observe data -> fit -> inference.

Only caller-invoked transformer.transform is executed by the recorder.
Audit never trains, transforms, predicts, repairs, or chooses a model.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .fit_provenance import _hashes, _matrix, _parse_records as _fit_records
from .output_provenance import _parse_records as _output_records

_SHA = re.compile(r"^[0-9a-f]{64}$")
Status = Literal["ok", "problem", "unknown", "not_applicable"]


@dataclass(frozen=True)
class TransformCallRecord:
    split: Literal["train", "evaluation"]
    input_space: Literal["raw", "transformed"]
    transformer_type: str
    transformer_tag: str
    row_ids: tuple[str, ...]
    input_width: int
    input_row_sha256: tuple[str, ...]
    input_batch_sha256: str
    output_status: Literal["witnessed", "unavailable"]
    output_width: int | None
    output_row_sha256: tuple[str, ...] | None
    output_batch_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransformCheck:
    code: str
    status: Status
    detail: str


def _rehash(row_hashes: tuple[str, ...] | list[str], width: int) -> str:
    data = b"FXAI-BATCH-V1" + len(row_hashes).to_bytes(8, "big") + width.to_bytes(8, "big")
    return hashlib.sha256(data + b"".join(bytes.fromhex(v) for v in row_hashes)).hexdigest()


class TransformRecorder:
    """Opt-in wrapper preserving the caller's unmodified transform result."""

    def __init__(self) -> None:
        self._records: list[TransformCallRecord] = []

    @property
    def records(self) -> tuple[TransformCallRecord, ...]:
        return tuple(self._records)

    def transform(
        self,
        transformer: Any,
        X: Any,
        *,
        split: Literal["train", "evaluation"],
        row_ids: tuple[str, ...],
        transformer_tag: str = "preprocessor",
        input_space: Literal["raw", "transformed"] = "raw",
        **kwargs: Any,
    ) -> Any:
        if split not in ("train", "evaluation") or input_space not in ("raw", "transformed"):
            raise ValueError("Unsupported split or feature space")
        if not isinstance(transformer_tag, str) or not transformer_tag:
            raise ValueError("A nonempty transformer tag is required")
        source = _matrix(X)
        ids = tuple(row_ids)
        if len(ids) != len(source) or len(set(ids)) != len(ids) or any(not isinstance(v, str) or not v for v in ids):
            raise ValueError("Transform input row IDs must be unique and aligned")
        input_rows, input_batch = _hashes(source)
        result = transformer.transform(X, **kwargs)  # caller explicitly initiates this call
        try:
            output = _matrix(result)
            if len(output) != len(source):
                raise ValueError("Row counts differ")
        except (TypeError, ValueError, OverflowError):
            state: Literal["witnessed", "unavailable"] = "unavailable"
            output_rows = None
            output_batch = None
            output_width = None
        else:
            state = "witnessed"
            output_rows, output_batch = _hashes(output)
            output_width = int(output.shape[1])
        self._records.append(
            TransformCallRecord(
                split=split, input_space=input_space,
                transformer_type=type(transformer).__name__, transformer_tag=transformer_tag,
                row_ids=ids, input_width=int(source.shape[1]), input_row_sha256=input_rows,
                input_batch_sha256=input_batch, output_status=state,
                output_width=output_width, output_row_sha256=output_rows,
                output_batch_sha256=output_batch,
            )
        )
        return result


def _parse(raw: Any) -> tuple[TransformCallRecord, ...]:
    if not isinstance(raw, (tuple, list)):
        raise ValueError("Expected transform trace array")
    found = []
    for value in raw:
        try:
            r = value if isinstance(value, TransformCallRecord) else TransformCallRecord(**value)
            if r.split not in ("train", "evaluation") or r.input_space not in ("raw", "transformed"):
                raise ValueError("Invalid split or space")
            if not isinstance(r.transformer_type, str) or not r.transformer_type or not isinstance(r.transformer_tag, str) or not r.transformer_tag:
                raise ValueError("Invalid declared transformer")
            if not isinstance(r.input_width, int) or isinstance(r.input_width, bool) or r.input_width < 1:
                raise ValueError("Invalid feature width")
            if not isinstance(r.row_ids, (list, tuple)) or not r.row_ids or len(r.row_ids) != len(set(r.row_ids)):
                raise ValueError("Invalid row identity")
            if any(not isinstance(v, str) or not v for v in r.row_ids):
                raise ValueError("Row IDs must be nonempty strings")
            if not isinstance(r.input_row_sha256, (list, tuple)) or len(r.row_ids) != len(r.input_row_sha256):
                raise ValueError("Invalid input hashes")
            if any(not isinstance(v, str) or not _SHA.fullmatch(v) for v in r.input_row_sha256):
                raise ValueError("Invalid input hash value")
            if not isinstance(r.input_batch_sha256, str) or not _SHA.fullmatch(r.input_batch_sha256) or _rehash(r.input_row_sha256, r.input_width) != r.input_batch_sha256:
                raise ValueError("Inconsistent input batch")
            if r.output_status == "witnessed":
                if not isinstance(r.output_width, int) or isinstance(r.output_width, bool) or r.output_width < 1:
                    raise ValueError("Invalid output width")
                if not isinstance(r.output_row_sha256, (list, tuple)) or len(r.output_row_sha256) != len(r.row_ids):
                    raise ValueError("Invalid output hashes")
                if any(not isinstance(v, str) or not _SHA.fullmatch(v) for v in r.output_row_sha256):
                    raise ValueError("Invalid output hash values")
                if not isinstance(r.output_batch_sha256, str) or not _SHA.fullmatch(r.output_batch_sha256) or _rehash(r.output_row_sha256, r.output_width) != r.output_batch_sha256:
                    raise ValueError("Inconsistent output batch")
            elif r.output_status != "unavailable" or any(v is not None for v in (r.output_width, r.output_row_sha256, r.output_batch_sha256)):
                raise ValueError("Unavailable outputs must have no digest")
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            raise ValueError("Malformed transform record") from exc
        found.append(r)
    return tuple(found)


def audit_transform_chain(
    raw: Any,
    *,
    train_features: Any | None = None,
    evaluation_features: Any | None = None,
    train_ids: tuple[str, ...] | None = None,
    evaluation_ids: tuple[str, ...] | None = None,
    fit_records: Any | None = None,
    output_records: Any | None = None,
    preprocessor_applied: bool | None = None,
    task: str | None = None,
) -> tuple[TransformCheck, ...]:
    """One dense transform, one train/eval batch; checks local trace consistency."""
    if raw is None or (isinstance(raw, (list, tuple)) and not raw):
        state: Status = "not_applicable" if preprocessor_applied is False else "unknown"
        return (TransformCheck("transform_trace", state, "No witnessed transformation"),)
    if preprocessor_applied is False:
        return (TransformCheck("transform_trace", "problem", "Transforms contradict declared absence of preprocessing"),)
    try:
        records = _parse(raw)
    except ValueError:
        return (TransformCheck("transform_trace", "problem", "Invalid transform trace"),)
    checks: list[TransformCheck] = []
    by_split: dict[str, TransformCallRecord] = {}
    source_state: dict[str, Status] = {}
    for split, features, ids in (("train", train_features, train_ids), ("evaluation", evaluation_features, evaluation_ids)):
        code = f"transform_{split}_input"
        matches = [record for record in records if record.split == split]
        if len(matches) != 1:
            checks.append(TransformCheck(code, "unknown", "Single unambiguous transform call required"))
            source_state[split] = "unknown"
            continue
        record = matches[0]
        by_split[split] = record
        if record.input_space != "raw" or features is None or ids is None:
            checks.append(TransformCheck(code, "unknown", "Raw source features and IDs required; multiple chained transforms unsupported"))
            source_state[split] = "unknown"
            continue
        try:
            source = _matrix(features)
        except (ValueError, TypeError, OverflowError):
            checks.append(TransformCheck(code, "unknown", "Cannot inspect raw source feature values"))
            source_state[split] = "unknown"
            continue
        _, expected = _hashes(source)
        mismatch = (
            tuple(record.row_ids) != tuple(ids)
            or record.input_width != source.shape[1]
            or record.input_batch_sha256 != expected
        )
        state = "problem" if mismatch else "ok"
        source_state[split] = state
        checks.append(TransformCheck(code, state, "Mismatch of source rows or transform input values" if mismatch else "Transform inputs match declared source rows"))
    if len(by_split) == 2:
        tr, ev = by_split["train"], by_split["evaluation"]
        different = tr.transformer_type != ev.transformer_type or tr.transformer_tag != ev.transformer_tag
        checks.append(TransformCheck("transform_identity", "problem" if different else "ok", "Declared transformer type/tag differ" if different else "Matching declared transformer type/tag (object identity not authenticated)"))
    else:
        checks.append(TransformCheck("transform_identity", "unknown", "Both split records required"))

    try:
        fits = _fit_records(fit_records) if fit_records is not None else None
    except (ValueError, TypeError):
        fits = None
    tr = by_split.get("train")
    if tr is None or source_state.get("train") != "ok" or fits is None:
        checks.append(TransformCheck("transform_preprocessor_fit_link", "unknown", "Source-bound preprocessor fit evidence unavailable"))
    else:
        matched = [item for item in fits if item.role == "preprocessor"]
        if len(matched) != 1:
            checks.append(TransformCheck("transform_preprocessor_fit_link", "unknown", "Exactly one preprocessor fit call required"))
        else:
            fit = matched[0]
            mismatch = (
                fit.feature_space != "raw" or tuple(fit.row_ids) != tuple(tr.row_ids)
                or fit.batch_sha256 != tr.input_batch_sha256 or fit.estimator_type != tr.transformer_type
            )
            checks.append(TransformCheck("transform_preprocessor_fit_link", "problem" if mismatch else "ok", "Preprocessor fit source/type differs from transform" if mismatch else "Fit and transform source/type agree"))
    if tr is None or source_state.get("train") != "ok" or tr.output_status != "witnessed" or fits is None:
        checks.append(TransformCheck("transform_model_fit_link", "unknown", "Source-bound transformed training output or model fit unavailable"))
    else:
        matched = [item for item in fits if item.role == "model"]
        if len(matched) != 1 or matched[0].feature_space != "transformed":
            checks.append(TransformCheck("transform_model_fit_link", "unknown", "One transformed model fit required"))
        else:
            fit = matched[0]
            mismatch = (
                tuple(fit.row_ids) != tuple(tr.row_ids)
                or fit.n_features != tr.output_width
                or fit.batch_sha256 != tr.output_batch_sha256
            )
            checks.append(TransformCheck("transform_model_fit_link", "problem" if mismatch else "ok", "Fitted model saw a different transformed train batch" if mismatch else "Fitted model input matches transformed train batch"))

    try:
        outputs = _output_records(output_records) if output_records is not None else None
    except (ValueError, TypeError):
        outputs = None
    ev = by_split.get("evaluation")
    for kind in ("predict", "predict_proba"):
        code = f"transform_{kind}_link"
        if kind == "predict_proba" and task in ("regression", "clustering"):
            checks.append(TransformCheck(code, "not_applicable", "No classification probability channel"))
        elif ev is None or source_state.get("evaluation") != "ok" or ev.output_status != "witnessed" or outputs is None:
            checks.append(TransformCheck(code, "unknown", "Source-bound transformed evaluation result or output call unavailable"))
        else:
            matched = [item for item in outputs if item.kind == kind]
            if len(matched) != 1 or matched[0].feature_space != "transformed":
                checks.append(TransformCheck(code, "unknown", "Exactly one transformed inference call required"))
            else:
                event = matched[0]
                mismatch = (
                    tuple(event.row_ids) != tuple(ev.row_ids)
                    or event.n_features != ev.output_width
                    or event.input_batch_sha256 != ev.output_batch_sha256
                )
                checks.append(TransformCheck(code, "problem" if mismatch else "ok", "Inference input disagrees with transformed evaluation output" if mismatch else "Inference input matches transformed evaluation output"))
    return tuple(checks)
