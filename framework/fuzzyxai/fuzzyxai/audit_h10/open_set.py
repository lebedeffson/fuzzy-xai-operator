from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .family_classifier import FaultFamilyClassifier
from .features import extract_feature_dict, vectorize
from .models import RouteObservation


@dataclass
class UnknownFaultDetector:
    threshold_known: float = 0.50
    threshold_anomaly: float = 1.0

    def fit(self, routes: list[RouteObservation], classifier: FaultFamilyClassifier) -> "UnknownFaultDetector":
        names = classifier.feature_names
        rows = np.vstack([vectorize(extract_feature_dict(route), names) for route in routes])
        self.center_ = np.mean(rows, axis=0)
        self.scale_ = np.maximum(np.std(rows, axis=0), 0.05)
        return self

    def anomaly_score(self, route: RouteObservation, classifier: FaultFamilyClassifier) -> float:
        if not hasattr(self, "center_"):
            return 0.0
        value = vectorize(extract_feature_dict(route), classifier.feature_names)
        z = (value - self.center_) / self.scale_
        return float(np.sqrt(np.mean(np.square(z))))

    def evaluate(self, route: RouteObservation, classifier: FaultFamilyClassifier) -> tuple[bool, float, float]:
        probabilities = classifier.probabilities(route)
        known_confidence = max(probabilities.values(), default=0.0)
        anomaly = self.anomaly_score(route, classifier)
        return known_confidence < self.threshold_known and anomaly > self.threshold_anomaly, known_confidence, anomaly
