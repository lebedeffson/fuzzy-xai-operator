"""Public structural-diagnostics API.

The package diagnoses route integrity. It does not infer model correctness or
domain safety from a structural violation.
"""

from .causes import DiagnosticCauseAnalyzer
from .compat import LEGACY_POLICY_WARNING, diagnose_h10_observation
from .contract_registry import ContractCheck, ContractRegistry
from .contracts import (
    BatchDiagnosticReport,
    CauseStatement,
    Contract,
    DiagnosticCut,
    DiagnosticIssue,
    DiagnosticReport,
    RecertificationReport,
    RepairCostModel,
    RepairExecutionContext,
    RepairPlan,
    RepairStep,
    RouteEdge,
    RouteGraph,
    RouteNode,
    StepExecutionResult,
    ValidationObligation,
    ValidationResult,
)
from .minimal_cut import MinimalDiagnosticCutFinder, verify_cut
from .recertification import RouteRecertifier
from .repair_executor import RepairExecutor
from .repair_planner import ActionableRepairPlanner
from .repair_registry import RepairProvider, RepairProviderRegistry
from .reporter import DiagnosticReporter
from .route_graph import RouteGraphBuilder
from .service import DiagnosticService, diagnose_route
from .validator import DiagnosticValidator

__all__ = [
    "ActionableRepairPlanner",
    "BatchDiagnosticReport",
    "CauseStatement",
    "Contract",
    "ContractCheck",
    "ContractRegistry",
    "DiagnosticCauseAnalyzer",
    "DiagnosticCut",
    "DiagnosticIssue",
    "DiagnosticReport",
    "DiagnosticReporter",
    "DiagnosticService",
    "DiagnosticValidator",
    "MinimalDiagnosticCutFinder",
    "LEGACY_POLICY_WARNING",
    "RecertificationReport",
    "RepairCostModel",
    "RepairExecutionContext",
    "RepairExecutor",
    "RepairPlan",
    "RepairProvider",
    "RepairProviderRegistry",
    "RepairStep",
    "RouteEdge",
    "RouteGraph",
    "RouteGraphBuilder",
    "RouteNode",
    "RouteRecertifier",
    "StepExecutionResult",
    "ValidationObligation",
    "ValidationResult",
    "diagnose_route",
    "diagnose_h10_observation",
    "verify_cut",
]
