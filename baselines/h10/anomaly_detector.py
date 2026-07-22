from __future__ import annotations

import numpy as np

from .common import BaselineResult, changed, fields, missing


class AnomalyDetectorBaseline:
    name = "anomaly_detector"

    def __init__(self, threshold: float = 0.20) -> None:
        self.threshold = threshold

    def diagnose(self, route: object) -> BaselineResult:
        names = fields(route)
        active = tuple(dict.fromkeys(missing(route) + changed(route)))
        score = float(np.mean([field in active for field in names])) if names else 0.0
        if score <= self.threshold:
            return BaselineResult("valid", confidence=1.0 - score, anomaly_score=score)
        return BaselineResult("invalid", unknown=True, abstained=True, confidence=0.0, anomaly_score=score)
