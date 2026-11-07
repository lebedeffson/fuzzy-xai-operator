"""Interval type-2 (IT2) fuzzy inference with auditable uncertainty propagation.

This module deliberately distinguishes three different quantities:
  * interval type-2 antecedent membership grades (LMF/UMF, the FOU),
  * the consequent output range induced by uncertain fuzzy thresholds, and
  * a point estimate from an explicitly declared midpoint defuzzification.

An induced output interval is NOT a calibrated predictive confidence interval.
The native additive fuzzy-logit architecture uses paired, coupled low/high
memberships. Its exact logit bounds can be computed analytically; the classic
Karnik-Mendel center-of-sets algorithm applies to a different architecture
(normalized firing strengths with crisp singleton consequents) and is included
as an independent reference operation.

No external libraries are needed to replay a saved model manifest and proof.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "fuzzyxai-native-it2-v1"
ACTIONS = ("accept", "lower_confidence", "request_more_data", "defer_to_human", "block")


def _finite(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def saturated_high(x: float, left: float, right: float) -> float:
    """Right-shoulder type-1 membership; constant outside its fitted support."""
    x, left, right = (_finite(x, "x"), _finite(left, "left"), _finite(right, "right"))
    if not left < right:
        raise ValueError("membership requires left < right")
    return min(1.0, max(0.0, (x - left) / (right - left)))


def sigmoid(z: float) -> float:
    z = _finite(z, "logit")
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


@dataclass(frozen=True)
class MembershipInterval:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        lo, hi = _finite(self.lower, "membership lower"), _finite(self.upper, "membership upper")
        if not 0 <= lo <= hi <= 1:
            raise ValueError("membership bounds must satisfy 0 <= LMF <= UMF <= 1")

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True)
class IT2Shoulder:
    """Type-2 shoulder using an empirical rectangular set of uncertain knots.

    The knots are correlated when estimated from the same bootstrap draw.
    Permitting all combinations of their marginal ranges is conservative: the
    resulting FOU encloses those bootstrap configurations, not necessarily the
    tight FOU of the joint bootstrap distribution. No statistical coverage
    interpretation is attached to this parameter rectangle.
    """

    left_min: float
    left_max: float
    right_min: float
    right_max: float
    nominal_left: float
    nominal_right: float

    def __post_init__(self) -> None:
        bounds = [
            _finite(self.left_min, "left_min"),
            _finite(self.left_max, "left_max"),
            _finite(self.right_min, "right_min"),
            _finite(self.right_max, "right_max"),
            _finite(self.nominal_left, "nominal_left"),
            _finite(self.nominal_right, "nominal_right"),
        ]
        a, b, c, d, n_lo, n_hi = bounds
        if not a <= n_lo <= b < c <= n_hi <= d:
            raise ValueError("require left_min <= nominal_left <= left_max < right_min <= nominal_right <= right_max")

    def high(self, value: float) -> MembershipInterval:
        # Saturated high(x; left,right) is nonincreasing in each knot.
        # Hence opposite corners of the knot rectangle give the exact extrema.
        lower = saturated_high(value, self.left_max, self.right_max)
        upper = saturated_high(value, self.left_min, self.right_min)
        return MembershipInterval(lower=lower, upper=upper)

    def low(self, value: float) -> MembershipInterval:
        high = self.high(value)
        return MembershipInterval(lower=1.0 - high.upper, upper=1.0 - high.lower)

    def nominal_high(self, value: float) -> float:
        return saturated_high(value, self.nominal_left, self.nominal_right)


@dataclass(frozen=True)
class IT2FeatureRule:
    feature_index: int
    feature_name: str
    low_coefficient: float
    high_coefficient: float
    membership: IT2Shoulder

    def __post_init__(self) -> None:
        if isinstance(self.feature_index, bool) or not isinstance(self.feature_index, int) or self.feature_index < 0:
            raise ValueError("feature_index must be a nonnegative integer")
        if not self.feature_name.strip():
            raise ValueError("feature_name must be present")
        _finite(self.low_coefficient, "low coefficient")
        _finite(self.high_coefficient, "high coefficient")


@dataclass(frozen=True)
class IT2FuzzyLogit:
    """Native paired-rule logistic model with rectangular IT2 parameter FOU."""

    intercept: float
    n_input_features: int
    features: tuple[IT2FeatureRule, ...]
    source_fit: str = "train_only_bootstrap_quantiles"

    def __post_init__(self) -> None:
        _finite(self.intercept, "intercept")
        if isinstance(self.n_input_features, bool) or not isinstance(self.n_input_features, int) or self.n_input_features < 1:
            raise ValueError("n_input_features must be positive")
        if not self.features:
            raise ValueError("at least one genuine fuzzy feature is needed")
        indices = [f.feature_index for f in self.features]
        if len(indices) != len(set(indices)) or any(index >= self.n_input_features for index in indices):
            raise ValueError("feature indices must be unique and within the input schema")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IT2FuzzyLogit:
        return cls(
            intercept=_finite(payload["intercept"], "intercept"),
            n_input_features=payload["n_input_features"],
            features=tuple(
                IT2FeatureRule(
                    feature_index=f["feature_index"],
                    feature_name=f["feature_name"],
                    low_coefficient=_finite(f["low_coefficient"], "low_coefficient"),
                    high_coefficient=_finite(f["high_coefficient"], "high_coefficient"),
                    membership=IT2Shoulder(**f["membership"]),
                ) for f in payload["features"]
            ),
            source_fit=str(payload.get("source_fit", "train_only_bootstrap_quantiles")),
        )

    def infer(self, input_row: Sequence[float]) -> dict[str, Any]:
        if len(input_row) != self.n_input_features:
            raise ValueError("input shape differs from the recorded model schema")
        rows = [_finite(value, f"input[{i}]") for i, value in enumerate(input_row)]
        min_logit = max_logit = nominal_logit = self.intercept
        evidence: list[dict[str, Any]] = []
        for rule in self.features:
            value = rows[rule.feature_index]
            low = rule.membership.low(value)
            high = rule.membership.high(value)
            nominal = rule.membership.nominal_high(value)
            # The two terms are *coupled*: mu_low = 1 - mu_high for each
            # possible pair of knots. Treating them as independent would
            # over-estimate output uncertainty and be mathematically looser.
            contrast = rule.high_coefficient - rule.low_coefficient
            c_lower = rule.low_coefficient + min(contrast * high.lower, contrast * high.upper)
            c_upper = rule.low_coefficient + max(contrast * high.lower, contrast * high.upper)
            c_nominal = rule.low_coefficient + contrast * nominal
            min_logit += c_lower
            max_logit += c_upper
            nominal_logit += c_nominal
            evidence.append({
                "feature_index": rule.feature_index,
                "feature_name": rule.feature_name,
                "observed_value": value,
                "low_term": asdict(low),
                "high_term": asdict(high),
                "high_term_nominal": nominal,
                "logit_contribution_lower": c_lower,
                "logit_contribution_upper": c_upper,
                "logit_contribution_nominal": c_nominal,
                "coupled_antecedents": True,
            })
        if not min_logit - 1e-10 <= nominal_logit <= max_logit + 1e-10:
            raise AssertionError("nominal inference escaped the induced IT2 bounds")
        p_lower, p_upper = sigmoid(min_logit), sigmoid(max_logit)
        return {
            "representation": "interval_type_2_FOU",
            "inference_architecture": "paired_coupled_logistic_rules",
            "type_reduction": "analytic_interval_logit_extrema_then_sigmoid",
            "defuzzification": "mean_probability_endpoints_research_policy",
            "logit_lower": min_logit,
            "logit_upper": max_logit,
            "nominal_logit": nominal_logit,
            "probability_lower": p_lower,
            "probability_upper": p_upper,
            "probability_nominal_type1": sigmoid(nominal_logit),
            "probability_midpoint": (p_lower + p_upper) / 2.0,
            "induced_width": p_upper - p_lower,
            "feature_evidence": evidence,
            "limitations": [
                "FOU parameter bounds come from exploratory training-only bootstrap quantiles",
                "Output bounds are parameter-induced, NOT calibrated predictive confidence intervals",
                "Decision abstention is a research policy, not a clinical treatment rule",
            ],
        }


def robust_interval_decision(inference: Mapping[str, Any], *, threshold: float = 0.5,
                             max_width: float = 0.20, missing_source: bool = False) -> dict[str, Any]:
    threshold, max_width = _finite(threshold, "decision threshold"), _finite(max_width, "maximum interval width")
    if not 0 < threshold < 1 or not 0 <= max_width <= 1:
        raise ValueError("decision threshold and max width must be in (0,1) and [0,1]")
    lo = _finite(inference["probability_lower"], "lower probability")
    hi = _finite(inference["probability_upper"], "upper probability")
    if not 0 <= lo <= hi <= 1:
        raise ValueError("model output is not an interval in [0,1]")
    if missing_source:
        action, reason = "block", "a required input was missing; no verified source"
    elif lo < threshold <= hi:
        action, reason = "request_more_data", "interval intersects the predefined classification boundary"
    elif hi - lo > max_width:
        action, reason = "defer_to_human", "parameter-induced output interval exceeds the declared width limit"
    else:
        action, reason = "accept", "entire induced interval lies on one side of the decision boundary"
    return {
        "action": action,
        "reason": reason,
        "threshold": threshold,
        "max_width": max_width,
        "accepted": action == "accept",
        "nominal_predicted_class": int(float(inference["probability_nominal_type1"]) >= threshold),
        "robust_predicted_class": (int(lo >= threshold) if action == "accept" else None),
    }


def make_it2_receipt(model: IT2FuzzyLogit, input_row: Sequence[float], *,
                     source_id: str, dataset_version: str, threshold: float = 0.5,
                     max_width: float = 0.20, expected_model_sha256: str | None = None) -> dict[str, Any]:
    """Reproducible numeric receipt; not a cryptographic proof of who fitted the model."""
    inference = model.infer(input_row)
    manifest = model.to_dict()
    model_hash = content_sha256(manifest)
    if expected_model_sha256 is not None and model_hash != expected_model_sha256:
        raise ValueError("model manifest does not match the trusted model anchor")
    if not source_id.strip() or not dataset_version.strip():
        raise ValueError("source identity and dataset version are required")
    core = {
        "schema_version": SCHEMA_VERSION,
        "scope": "numeric_type2_rule_path_not_predictive_or_clinical_validation",
        "source_id": source_id,
        "dataset_version": dataset_version,
        "model_manifest": manifest,
        "model_sha256": model_hash,
        "input_values": list(input_row),
        "input_sha256": content_sha256(list(input_row)),
        "inference": inference,
        "decision": robust_interval_decision(inference, threshold=threshold, max_width=max_width),
    }
    return {**core, "receipt_sha256": content_sha256(core)}


def verify_it2_receipt(receipt: Mapping[str, Any], *, trusted_model_sha256: str | None = None) -> dict[str, Any]:
    """Recompute all reported numbers, policy, FOU and hashes from manifest."""
    errors: list[str] = []
    try:
        if receipt.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unknown type-2 receipt schema")
        model = IT2FuzzyLogit.from_dict(receipt["model_manifest"])
        model_hash = content_sha256(model.to_dict())
        if model_hash != receipt.get("model_sha256"):
            raise ValueError("model fingerprint mismatch")
        if trusted_model_sha256 is not None and model_hash != trusted_model_sha256:
            raise ValueError("model does not match the external trusted fingerprint")
        values = receipt["input_values"]
        if content_sha256(values) != receipt.get("input_sha256"):
            raise ValueError("input hash mismatch")
        policy = receipt["decision"]
        if not isinstance(policy, Mapping):
            raise ValueError("invalid recorded policy")
        expected = make_it2_receipt(
            model,
            values,
            source_id=receipt["source_id"],
            dataset_version=receipt["dataset_version"],
            threshold=_finite(policy["threshold"], "threshold"),
            max_width=_finite(policy["max_width"], "max_width"),
        )
        for key in ("inference", "decision", "model_manifest", "model_sha256", "input_sha256", "receipt_sha256"):
            if _canonical_bytes(receipt.get(key)) != _canonical_bytes(expected[key]):
                errors.append(f"mismatch:{key}")
        if receipt.get("scope") != expected["scope"]:
            errors.append("mismatch:scope")
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
        errors.append(str(exc))
    return {"valid": not errors, "errors": errors, "schema_version": SCHEMA_VERSION}


def karnik_mendel_center_of_sets(consequents: Sequence[float], firing_lower: Sequence[float],
                                  firing_upper: Sequence[float], *, tolerance: float = 1e-12,
                                  max_iterations: int = 1000) -> tuple[float, float]:
    """Classic KM center-of-sets for singleton consequents and interval firings.

    This normalized inference operation is NOT used by `IT2FuzzyLogit`, whose
    additive paired-rule output has an exact analytic interval bound.
    """
    if not consequents or len(consequents) != len(firing_lower) or len(consequents) != len(firing_upper):
        raise ValueError("non-empty, equal-length consequents and interval firing vectors required")
    rows = sorted(zip(consequents, firing_lower, firing_upper), key=lambda item: float(item[0]))
    values = [_finite(row[0], "consequent") for row in rows]
    low = [_finite(row[1], "firing lower") for row in rows]
    high = [_finite(row[2], "firing upper") for row in rows]
    if any(not 0 <= left <= right <= 1 for left, right in zip(low, high)) or sum(high) <= 0:
        raise ValueError("require 0 <= lower <= upper <= 1 and positive possible firing mass")
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("invalid KM iteration controls")

    def solve(side: str) -> float:
        y = sum(v * w for v, w in zip(values, high)) / sum(high)
        for _ in range(max_iterations):
            if side == "left":
                chosen = [h if c <= y else l for c, l, h in zip(values, low, high)]
            else:
                chosen = [l if c <= y else h for c, l, h in zip(values, low, high)]
            total = sum(chosen)
            if total <= 0:
                # Possible only in a degenerate case; the boundary of the
                # reachable nonzero-firing supports is still well-defined.
                active = [c for c, h in zip(values, high) if h > 0]
                return min(active) if side == "left" else max(active)
            next_y = sum(c * w for c, w in zip(values, chosen)) / total
            if abs(next_y - y) <= tolerance:
                return next_y
            y = next_y
        raise ArithmeticError("KM switch point failed to converge")

    return (solve("left"), solve("right"))
