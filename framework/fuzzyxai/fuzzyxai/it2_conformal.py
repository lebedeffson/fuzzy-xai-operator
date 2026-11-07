"""Optional split-conformal prediction sets for an already fitted IT2 classifier.

Split conformal and fuzzy footprints of uncertainty answer different questions:
FOU bounds quantify membership-parameter variation; conformal sets provide
conditional-on-assumptions marginal label coverage under exchangeability.
Never infer per-patient, clinical, conditional, or distribution-shift coverage.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .interval_type2 import IT2FuzzyLogit, content_sha256

_SCHEMA = "fuzzyxai-it2-split-conformal-LAC-v1"
_SHA = re.compile(r"^[0-9a-fA-F]{64}$")


def _finite_unit(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} cannot be bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite in [0,1]") from exc
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be finite in [0,1]")
    return result


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("calibration/training object IDs must be non-empty strings")
    return _sha256("fuzzyxai:conformal:id:v1:" + value)


def _quantile(scores: Sequence[float], alpha: float) -> float | None:
    # Finite-sample split conformal correction: k=ceil((n+1)*(1-alpha)).
    # k=n+1 means an infinite threshold (both labels included), serialized as None.
    n = len(scores)
    if n == 0:
        raise ValueError("calibration scores cannot be empty")
    k = math.ceil((n + 1) * (1.0 - alpha))
    return None if k > n else float(sorted(scores)[k - 1])


@dataclass(frozen=True)
class IT2SplitConformal:
    """Frozen LAC calibration manifest; scores are NOT the training examples."""

    alpha: float
    model_sha256: str
    calibration_dataset_version: str
    calibration_scores: tuple[float, ...]
    calibration_id_hashes: tuple[str, ...]
    training_id_hashes: tuple[str, ...]
    threshold: float | None
    receipt_sha256: str
    method: str = _SCHEMA

    @classmethod
    def fit_probabilities(
        cls,
        probabilities_positive: Sequence[float],
        labels: Sequence[int],
        *,
        model_sha256: str,
        calibration_ids: Sequence[str],
        training_ids: Sequence[str],
        calibration_dataset_version: str,
        alpha: float = 0.1,
    ) -> IT2SplitConformal:
        p = [_finite_unit(value, "positive class probability") for value in probabilities_positive]
        y = list(labels)
        if not p or len(p) != len(y) or len(p) != len(calibration_ids):
            raise ValueError("calibration probabilities, labels and IDs must have equal nonzero lengths")
        if any(type(v) is not int or v not in (0, 1) for v in y):
            raise ValueError("split conformal requires actual binary calibration labels 0/1")
        if not isinstance(model_sha256, str) or _SHA.fullmatch(model_sha256) is None:
            raise ValueError("model_sha256 must be a full SHA256 digest")
        if not isinstance(calibration_dataset_version, str) or not calibration_dataset_version.strip():
            raise ValueError("calibration dataset version must be present")
        if isinstance(alpha, bool) or not 0 < _finite_unit(alpha, "alpha") < 1:
            raise ValueError("alpha must be strictly within (0,1)")
        if not training_ids:
            raise ValueError("explicit training IDs required to test train/calibration disjointness")
        cal_hashes = tuple(sorted(_hash_id(v) for v in calibration_ids))
        training_hashes = tuple(sorted(_hash_id(v) for v in training_ids))
        if len(cal_hashes) != len(set(cal_hashes)):
            raise ValueError("calibration object IDs must be unique")
        if len(training_hashes) != len(set(training_hashes)):
            raise ValueError("training object IDs must be unique")
        if set(cal_hashes) & set(training_hashes):
            raise ValueError("training and calibration object identities overlap")
        scores = tuple(sorted(1.0 - prob if label == 1 else prob for prob, label in zip(p, y)))
        core = {
            "method": _SCHEMA,
            "alpha": float(alpha),
            "model_sha256": model_sha256.lower(),
            "calibration_dataset_version": calibration_dataset_version,
            "calibration_scores": list(scores),
            "calibration_id_hashes": list(cal_hashes),
            "training_id_hashes": list(training_hashes),
            "threshold": _quantile(scores, alpha),
        }
        return cls(
            alpha=float(alpha), model_sha256=model_sha256.lower(),
            calibration_dataset_version=calibration_dataset_version,
            calibration_scores=scores, calibration_id_hashes=cal_hashes,
            training_id_hashes=training_hashes, threshold=core["threshold"],
            receipt_sha256=content_sha256(core),
        )

    @classmethod
    def fit_it2(
        cls, model: IT2FuzzyLogit, rows: Sequence[Sequence[float]], labels: Sequence[int],
        *, calibration_ids: Sequence[str], training_ids: Sequence[str],
        calibration_dataset_version: str, alpha: float = 0.1,
    ) -> IT2SplitConformal:
        if not isinstance(model, IT2FuzzyLogit):
            raise TypeError("fit_it2 requires a trained IT2FuzzyLogit")
        return cls.fit_probabilities(
            [model.infer(row)["probability_midpoint"] for row in rows], labels,
            model_sha256=content_sha256(model.to_dict()),
            calibration_ids=calibration_ids, training_ids=training_ids,
            calibration_dataset_version=calibration_dataset_version, alpha=alpha,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "model_sha256": self.model_sha256,
            "calibration_dataset_version": self.calibration_dataset_version,
            "calibration_scores": list(self.calibration_scores),
            "calibration_id_hashes": list(self.calibration_id_hashes),
            "training_id_hashes": list(self.training_id_hashes),
            "threshold": self.threshold,
            "receipt_sha256": self.receipt_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IT2SplitConformal:
        if not isinstance(payload, Mapping):
            raise TypeError("conformal calibration manifest must be a mapping")
        try:
            value = cls(
                alpha=payload["alpha"], model_sha256=payload["model_sha256"],
                calibration_dataset_version=payload["calibration_dataset_version"],
                calibration_scores=tuple(payload["calibration_scores"]),
                calibration_id_hashes=tuple(payload["calibration_id_hashes"]),
                training_id_hashes=tuple(payload["training_id_hashes"]),
                threshold=payload["threshold"], receipt_sha256=payload["receipt_sha256"],
                method=payload["method"],
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("invalid conformal calibration manifest") from exc
        value.verify()
        return value

    def verify(self) -> None:
        if self.method != _SCHEMA:
            raise ValueError("unknown conformal LAC schema")
        if isinstance(self.alpha, bool) or not 0 < _finite_unit(self.alpha, "alpha") < 1:
            raise ValueError("alpha must be in (0,1)")
        if not isinstance(self.model_sha256, str) or _SHA.fullmatch(self.model_sha256) is None:
            raise ValueError("model fingerprint must be SHA256")
        if not self.calibration_dataset_version or not isinstance(self.calibration_dataset_version, str):
            raise ValueError("calibration dataset version missing")
        scores = tuple(_finite_unit(s, "calibration score") for s in self.calibration_scores)
        if not scores or tuple(sorted(scores)) != scores:
            raise ValueError("calibration scores must be present in sorted order")
        for group in (self.calibration_id_hashes, self.training_id_hashes):
            if not group or any(not isinstance(x, str) or _SHA.fullmatch(x) is None for x in group):
                raise ValueError("invalid source-identity hash set")
            if len(group) != len(set(group)) or tuple(sorted(group)) != group:
                raise ValueError("source identity hash sets must be sorted and unique")
        if len(self.calibration_id_hashes) != len(scores):
            raise ValueError("source-identity count differs from calibration scores")
        if set(self.calibration_id_hashes) & set(self.training_id_hashes):
            raise ValueError("training and calibration overlap")
        q = _quantile(scores, self.alpha)
        if q != self.threshold:
            raise ValueError("finite-sample conformal quantile was tampered with")
        core = self.to_dict()
        digest = core.pop("receipt_sha256")
        if not isinstance(digest, str) or digest != content_sha256(core):
            raise ValueError("conformal calibration receipt hash mismatch")

    def predict_set(self, probability_positive: float, *, object_id: str) -> dict[str, Any]:
        """Return a label set and abstention instruction, never a clinical diagnosis."""
        self.verify()
        identity = _hash_id(object_id)
        if identity in self.calibration_id_hashes or identity in self.training_id_hashes:
            raise ValueError("conformal test object overlaps the declared calibration/training split")
        p = _finite_unit(probability_positive, "positive probability")
        q = self.threshold
        labels = [0, 1] if q is None else [label for label, score in ((0, p), (1, 1 - p)) if score <= q]
        return {
            "object_id": object_id,
            "label_set": labels,
            "set_size": len(labels),
            "action": "accept" if len(labels) == 1 else "request_more_data",
            "calibration_count": len(self.calibration_scores),
            "nominal_miscoverage": self.alpha,
            "threshold": q,
            "model_sha256": self.model_sha256,
            "calibration_receipt_sha256": self.receipt_sha256,
            "assumptions": [
                "Model fixed before calibration and no train/calibration/test overlap",
                "Future test label and calibration scores are exchangeable",
                "Marginal rather than conditional coverage; no clinical guarantee",
                "LAC may produce empty sets, which trigger abstention",
            ],
        }
