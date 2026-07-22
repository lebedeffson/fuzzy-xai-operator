from __future__ import annotations

from .common import BaselineResult


class HashVersionBaseline:
    name = "hash_version"
    checked = ("artifact_sha256", "model_version", "explainer_version", "preprocessing_signature")

    def diagnose(self, route: object) -> BaselineResult:
        differences = tuple(field for field in self.checked if route.expected.get(field) != route.observed.get(field))
        if not differences:
            return BaselineResult("valid", confidence=1.0)
        return BaselineResult("invalid", source_nodes=differences, cut_nodes=differences, repair_nodes=differences, confidence=1.0)
