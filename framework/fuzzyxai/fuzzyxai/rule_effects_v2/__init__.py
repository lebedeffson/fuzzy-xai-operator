"""Separate predictive and structural rule-effect estimands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .certificate_effect import CertificateRuleEffect, certificate_effect
from .conditional import ConditionalRule, conditional_effect, within_stratum_resample
from .detectability import DetectabilityPoint, registered_detectability_grid, summarize_detectability
from .matched_controls import MatchedRuleSet, eligible_candidate, match_controls, specific_effect
from .nonrefit import RuleEffectEstimate, nonrefit_effect
from .refit import refit_effect


@dataclass(frozen=True)
class RuleEffectData:
    train_values: np.ndarray
    train_labels: np.ndarray
    test_values: np.ndarray
    test_labels: np.ndarray
    strata: Sequence[object]
    metric: Callable[[np.ndarray, np.ndarray], float]


def assess_rule_effect(
    rule: ConditionalRule,
    model: object,
    data: RuleEffectData,
    estimands: Sequence[str] = ("nonrefit", "refit", "conditional"),
) -> dict[str, RuleEffectEstimate]:
    allowed = {"nonrefit", "refit", "conditional", "certificate"}
    unknown = set(estimands) - allowed
    if unknown:
        raise ValueError(f"unknown estimands: {sorted(unknown)}")
    if "certificate" in estimands:
        raise ValueError("certificate estimand requires explicit before/after certificates via certificate_effect")
    result: dict[str, RuleEffectEstimate] = {}
    if "nonrefit" in estimands:
        result["nonrefit"] = nonrefit_effect(model, rule, data.test_values, data.test_labels, metric=data.metric)
    if "refit" in estimands:
        result["refit"] = refit_effect(model, rule, data.train_values, data.train_labels, data.test_values, data.test_labels, metric=data.metric)
    if "conditional" in estimands:
        result["conditional"] = conditional_effect(model, rule, data.test_values, data.test_labels, data.strata, metric=data.metric)
    return result


__all__ = [
    "CertificateRuleEffect",
    "ConditionalRule",
    "DetectabilityPoint",
    "MatchedRuleSet",
    "RuleEffectData",
    "RuleEffectEstimate",
    "assess_rule_effect",
    "certificate_effect",
    "conditional_effect",
    "eligible_candidate",
    "match_controls",
    "nonrefit_effect",
    "refit_effect",
    "registered_detectability_grid",
    "specific_effect",
    "summarize_detectability",
    "within_stratum_resample",
]
