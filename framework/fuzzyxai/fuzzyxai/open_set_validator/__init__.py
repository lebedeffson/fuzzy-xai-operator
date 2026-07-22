"""Typed open-set structural fault detection and abstention."""

from .baselines import AnomalyScore, fit_one_class_scores, graph_anomaly_score, mahalanobis_unknown, maximum_softmax_unknown
from .contracts import OpenSetAssessment, OpenSetOutcome, OpenSetTrainingRow, StructuralObservation
from .features import FEATURE_REGIONS, structural_observation
from .model import OpenSetValidatorSpec, assess_open_set, fit_open_set_validator

__all__ = [
    "AnomalyScore",
    "FEATURE_REGIONS",
    "OpenSetAssessment",
    "OpenSetOutcome",
    "OpenSetTrainingRow",
    "OpenSetValidatorSpec",
    "StructuralObservation",
    "assess_open_set",
    "fit_one_class_scores",
    "fit_open_set_validator",
    "graph_anomaly_score",
    "mahalanobis_unknown",
    "maximum_softmax_unknown",
    "structural_observation",
]
