from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FEATURE_NAMES, extract_feature_dict, softmax, vectorize
from .models import FaultPrediction, RouteObservation
from .taxonomy import SPEC_BY_LEAF


@dataclass
class FaultFamilyClassifier:
    leaf_abstention_threshold: float = 0.60

    def __post_init__(self) -> None:
        self.feature_names = FEATURE_NAMES + ("missing_fraction", "path_disagreement")
        self._centroids: dict[str, np.ndarray] = {}
        self._patterns: dict[str, tuple[frozenset[str], ...]] = {}

    @staticmethod
    def _active_fields(route: RouteObservation) -> frozenset[str]:
        features = extract_feature_dict(route)
        return frozenset(field for field in FEATURE_NAMES if features.get(field, 0.0) > 0.0)

    def fit(self, samples: list[tuple[RouteObservation, str]]) -> "FaultFamilyClassifier":
        grouped: dict[str, list[np.ndarray]] = {}
        patterns: dict[str, set[frozenset[str]]] = {}
        for route, leaf in samples:
            grouped.setdefault(leaf, []).append(vectorize(extract_feature_dict(route), self.feature_names))
            patterns.setdefault(leaf, set()).add(self._active_fields(route))
        self._centroids = {leaf: np.mean(rows, axis=0) for leaf, rows in grouped.items()}
        self._patterns = {leaf: tuple(sorted(rows, key=lambda item: (len(item), tuple(sorted(item))))) for leaf, rows in patterns.items()}
        return self

    @staticmethod
    def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
        union = left | right
        return len(left & right) / len(union) if union else 1.0

    def probabilities(self, route: RouteObservation) -> dict[str, float]:
        if not self._centroids:
            return {}
        value = vectorize(extract_feature_dict(route), self.feature_names)
        active = self._active_fields(route)
        leaves = tuple(sorted(self._centroids))
        scores = []
        for leaf in leaves:
            distance = float(np.linalg.norm(value - self._centroids[leaf]))
            pattern_similarity = max((self._jaccard(active, pattern) for pattern in self._patterns.get(leaf, ())), default=0.0)
            containment = max((len(active & pattern) / max(len(active), 1) for pattern in self._patterns.get(leaf, ())), default=0.0)
            scores.append(5.0 * pattern_similarity + 2.0 * containment - 0.5 * distance)
        probabilities = softmax(np.asarray(scores, dtype=float))
        return dict(zip(leaves, (float(item) for item in probabilities), strict=True))

    def predict(self, route: RouteObservation, *, unknown: bool = False) -> FaultPrediction:
        probabilities = self.probabilities(route)
        if unknown or not probabilities:
            return FaultPrediction(None, None, max(probabilities.values(), default=0.0), True, unknown)
        leaf, confidence = max(probabilities.items(), key=lambda item: (item[1], item[0]))
        parent = SPEC_BY_LEAF[leaf].parent
        abstained = confidence < self.leaf_abstention_threshold
        return FaultPrediction(parent, None if abstained else leaf, confidence, abstained, False)
