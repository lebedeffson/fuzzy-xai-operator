"""Minimal isolated-process reference-response signer for binary linear logits.

The worker loads a trusted, safe JSON export of a fitted StandardScaler +
LogisticRegression. It NEVER signs arbitrary caller-provided effect vectors.
The caller provides a signed source prediction and input/reference features;
the worker itself evaluates the model, verifies source hashes, probabilities
and model identity, then issues a second HMAC receipt.

A subprocess boundary and a private key file are not equivalent to separate
machine/user credentials or independent model-owner attestation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from fuzzyxai.core.certified_route import (
    SignedPrediction,
    sign_local_effect_witness,
)
from fuzzyxai.selective_observer.local_effects import (
    collect_linear_reference_effects_batched,
    hash_numeric_input,
)

_SCHEMA = "fuzzyxai-linear-binary-logit-model-v1"
_REQUEST_SCHEMA = "fuzzyxai-linear-response-request-v1"
_HEX = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON = 1024 * 1024
_MAX_FEATURES = 4096


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")


def _finite(v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError("non-finite or non-numeric model/input parameter")
    return float(v)


def _finite_array(values: Any, *, width: int | None = None) -> tuple[float, ...]:
    if not isinstance(values, list) or not 0 < len(values) <= _MAX_FEATURES:
        raise ValueError("invalid numeric vector")
    if width is not None and len(values) != width:
        raise ValueError("feature width mismatch")
    return tuple(_finite(v) for v in values)


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("invalid identity")
    return value


@dataclass(frozen=True)
class LinearBinaryModel:
    version: str
    target_class: str
    producer_id: str
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    model_sha256: str

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> LinearBinaryModel:
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema", "model_version", "target_class", "producer_id", "feature_names",
            "scaler_mean", "scaler_scale", "coefficients", "intercept", "classes",
        } or manifest["schema"] != _SCHEMA:
            raise ValueError("unknown or incomplete model manifest")
        names = manifest["feature_names"]
        if (not isinstance(names, list) or not 0 < len(names) <= _MAX_FEATURES
                or len(set(names)) != len(names)
                or any(not isinstance(x, str) or not x.strip() for x in names)):
            raise ValueError("invalid feature axis")
        if manifest["classes"] != [0, 1]:
            raise ValueError("only binary 0/1 model class order is supported")
        width = len(names)
        mean = _finite_array(manifest["scaler_mean"], width=width)
        scale = _finite_array(manifest["scaler_scale"], width=width)
        if any(x <= 0 for x in scale):
            raise ValueError("scaler scale must be positive")
        coefficients = _finite_array(manifest["coefficients"], width=width)
        intercept = _finite(manifest["intercept"])
        # Preserve the canonical upstream fitted-parameter digest contract.
        model_hash = sha256(_canonical({
            "coef": [list(coefficients)], "intercept": [intercept],
            "classes": [0, 1], "scaler_mean": list(mean),
            "scaler_scale": list(scale),
        })).hexdigest()
        return cls(
            version=_text(manifest["model_version"]),
            target_class=_text(manifest["target_class"]),
            producer_id=_text(manifest["producer_id"]),
            feature_names=tuple(names), mean=mean, scale=scale,
            coefficients=coefficients, intercept=intercept,
            model_sha256=model_hash,
        )

    def target_logit(self, row: tuple[float, ...]) -> float:
        if len(row) != len(self.feature_names):
            raise ValueError("model input width mismatch")
        value = self.intercept + math.fsum(
            (v - mean) / scale * weight
            for v, mean, scale, weight in zip(
                row, self.mean, self.scale, self.coefficients, strict=True,
            )
        )
        if not math.isfinite(value):
            raise ValueError("non-finite source model score")
        return value


def make_signed_response(
    model: LinearBinaryModel, record: dict[str, Any], secret: bytes,
) -> dict[str, Any]:
    """Compute the model probe from trusted model parameters, not caller effects."""
    if not isinstance(record, dict) or set(record) != {
        "schema", "source_prediction", "input_values", "reference_values",
    } or record["schema"] != _REQUEST_SCHEMA:
        raise ValueError("unsupported model-response request")
    raw_anchor = record["source_prediction"]
    try:
        anchor = SignedPrediction(**raw_anchor) if isinstance(raw_anchor, dict) else None
    except (TypeError, ValueError) as exc:
        raise ValueError("source prediction cannot be decoded") from exc
    if (anchor is None or not anchor.valid_shape()
            or not isinstance(anchor.signature, str)
            or not _HEX.fullmatch(anchor.signature)):
        raise ValueError("source prediction is not a valid signed record")
    if (anchor.model_version != model.version
            or anchor.event_id != model.target_class
            or anchor.model_sha256 != model.model_sha256):
        raise ValueError("source prediction does not match the trusted model")
    x = _finite_array(record["input_values"], width=len(model.feature_names))
    reference = _finite_array(record["reference_values"], width=len(model.feature_names))
    if reference != model.mean:
        raise ValueError("reference vector must equal the trusted training-mean reference")
    if hash_numeric_input(model.feature_names, x) != anchor.source_sha256:
        raise ValueError("input bytes do not match signed prediction")
    # Only the positive-class logit convention is supported. Bind signed
    # probability to the model actually evaluated, not merely to an input hash.
    logit = model.target_logit(x)
    prob = (1.0 / (1.0 + math.exp(-logit))) if logit >= 0 else (
        math.exp(logit) / (1.0 + math.exp(logit))
    )
    if abs(prob - float(anchor.score)) > 1e-8:
        raise ValueError("signed probability does not match the trusted model")
    effects = collect_linear_reference_effects_batched(
        object_id=anchor.object_id, model_version=model.version,
        target_class=model.target_class, feature_names=model.feature_names,
        input_values=x, reference_values=reference,
        target_logits=lambda rows: tuple(model.target_logit(row) for row in rows),
    )
    witness = sign_local_effect_witness(
        effects, source_signature=anchor.signature,
        producer_id=model.producer_id, secret=secret,
    )
    return {
        "status": "signed_model_response", "schema": "fuzzyxai-linear-response-v1",
        "model_sha256": model.model_sha256,
        "model_effects": asdict(effects),
        "signed_effect_witness": asdict(witness),
    }


def _load_model(path: Path) -> LinearBinaryModel:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_JSON:
        raise ValueError("model manifest is missing or not a regular bounded file")
    content = path.read_bytes()
    data = json.loads(content, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")))
    return LinearBinaryModel.from_manifest(data)


def _load_signing_key(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("private signing key file not found")
    mode = path.stat().st_mode
    if os.name == "posix" and (mode & (stat.S_IRWXG | stat.S_IRWXO)):
        raise ValueError("private signing key file must not be group/world readable")
    secret = path.read_bytes()
    if not 32 <= len(secret) <= 4096:
        raise ValueError("signing secret must contain 32..4096 raw bytes")
    return secret


def _decode_request(payload: bytes) -> dict[str, Any]:
    if len(payload) > _MAX_JSON:
        raise ValueError("request too large")
    return json.loads(
        payload,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")),
    )


def _handle_request(model: LinearBinaryModel, payload: bytes, secret: bytes) -> tuple[dict[str, Any], int]:
    try:
        request = _decode_request(payload)
        return make_signed_response(model, request, secret), 0
    except (UnicodeError, TypeError, KeyError, ValueError, OverflowError, json.JSONDecodeError):
        return {"status": "insufficient_evidence", "code": "MODEL_PROBE_UNAVAILABLE"}, 3


def _output(response: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(response, sort_keys=True, allow_nan=False) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FuzzyXAI signed linear response worker")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--key-file-env", default="FUZZYXAI_PROBE_KEY_FILE")
    parser.add_argument("--stream", action="store_true", help="NDJSON requests; one response per line")
    parser.add_argument("--max-requests", type=int, default=10000)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.max_requests <= 100000:
            raise ValueError("invalid request limit")
        key_file = os.environ.get(args.key_file_env)
        if not key_file:
            raise ValueError("signing key not configured")
        secret = _load_signing_key(Path(key_file))
        model = _load_model(args.manifest)
    except (OSError, UnicodeError, TypeError, KeyError, ValueError, OverflowError, json.JSONDecodeError):
        _output({"status": "insufficient_evidence", "code": "MODEL_PROBE_UNAVAILABLE"})
        return 3
    if not args.stream:
        response, status = _handle_request(model, sys.stdin.buffer.read(_MAX_JSON + 1), secret)
        _output(response)
        return status
    # Long-lived local process; the model and signer key are loaded once.
    # Every message is bounded, failures abstain and never contaminate others.
    for _ in range(args.max_requests):
        payload = sys.stdin.buffer.readline(_MAX_JSON + 1)
        if not payload:
            break
        if len(payload) > _MAX_JSON:
            # Consume an oversized line in bounded chunks rather than parsing a tail.
            if not payload.endswith(b"\n"):
                while True:
                    chunk = sys.stdin.buffer.readline(_MAX_JSON + 1)
                    if not chunk or chunk.endswith(b"\n"):
                        break
            _output({"status": "insufficient_evidence", "code": "MODEL_PROBE_UNAVAILABLE"})
            continue
        response, _status = _handle_request(model, payload, secret)
        _output(response)
    # The stream transport exits successfully; clients check EACH response.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
