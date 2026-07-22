from __future__ import annotations

from .models import FaultPrediction, RouteObservation
from .route_validity import ValidityResult
from .structural_inference import infer_specs


class SourceLocalizer:
    def localize(
        self,
        route: RouteObservation,
        validity: ValidityResult,
        fault: FaultPrediction | None = None,
    ) -> tuple[str, ...]:
        fields = tuple(dict.fromkeys(validity.missing_fields + validity.mismatched_fields))
        specs = infer_specs(fields, fault)
        return tuple(dict.fromkeys(node for spec in specs for node in spec.source_nodes))
