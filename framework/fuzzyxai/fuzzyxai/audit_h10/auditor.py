from __future__ import annotations

from dataclasses import dataclass

from .diagnostic_cut import DiagnosticCutSolver
from .family_classifier import FaultFamilyClassifier
from .models import AuditDiagnosis, DiagnosticCutResult, FaultPrediction, RouteObservation
from .open_set import UnknownFaultDetector
from .recertification import execute_and_recertify
from .repair_planner import RepairSetPlanner
from .route_validity import validate_route
from .source_localizer import SourceLocalizer
from .trace import canonical_trace


@dataclass
class H10Auditor:
    classifier: FaultFamilyClassifier
    unknown_detector: UnknownFaultDetector
    cut_solver: DiagnosticCutSolver
    repair_planner: RepairSetPlanner
    source_localizer: SourceLocalizer

    @classmethod
    def create(cls, *, threshold_known: float = 0.50, threshold_anomaly: float = 1.0, leaf_threshold: float = 0.60) -> "H10Auditor":
        return cls(
            FaultFamilyClassifier(leaf_threshold),
            UnknownFaultDetector(threshold_known, threshold_anomaly),
            DiagnosticCutSolver(),
            RepairSetPlanner(),
            SourceLocalizer(),
        )

    def fit(self, samples: list[tuple[RouteObservation, str]]) -> "H10Auditor":
        self.classifier.fit(samples)
        self.unknown_detector.fit([route for route, _ in samples], self.classifier)
        return self

    def diagnose(self, route: RouteObservation, *, execute_repair: bool = True) -> AuditDiagnosis:
        validity = validate_route(route)
        if validity.status == "valid":
            fault = FaultPrediction(None, None, 1.0, False, False)
            cut = DiagnosticCutResult((), 0.0, True, "none", 0.0, 0)
            payload = {"route_id": route.route_id, "route_status": "valid", "fault": fault, "cut": cut, "repair_set": ()}
            trace = canonical_trace(payload)
            return AuditDiagnosis("valid", fault, (), cut, (), True, trace)
        if validity.status == "insufficient_evidence":
            unknown, known_confidence, anomaly = False, 0.0, 0.0
            fault = FaultPrediction(None, None, 0.0, True, False)
        else:
            unknown, known_confidence, anomaly = self.unknown_detector.evaluate(route, self.classifier)
            fault = self.classifier.predict(route, unknown=unknown)
        sources = self.source_localizer.localize(route, validity, fault)
        cut = self.cut_solver.solve(validity.invalid_paths, route.repair_costs)
        repairs = self.repair_planner.plan(cut, fault, validity)
        recertified = None
        if execute_repair and repairs:
            _, recertified = execute_and_recertify(route, repairs)
        payload = {
            "route_id": route.route_id,
            "route_status": validity.status,
            "fault": fault,
            "source_nodes": sources,
            "diagnostic_cut": {
                "cut_nodes": cut.cut_nodes,
                "total_cost": cut.total_cost,
                "optimal": cut.optimal,
                "solver": cut.solver,
                "covered_invalid_paths": cut.covered_invalid_paths,
            },
            "repair_set": repairs,
            "recertified": recertified,
            "known_confidence": known_confidence,
            "anomaly_score": anomaly,
        }
        return AuditDiagnosis(
            validity.status,
            fault,
            sources,
            cut,
            repairs,
            recertified,
            canonical_trace(payload),
            {"known_confidence": known_confidence, "anomaly_score": anomaly},
        )
