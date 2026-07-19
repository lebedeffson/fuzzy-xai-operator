"""Pre-registered Q1 empirical-remediation protocols.

This package is deliberately isolated from the defended operator core. It
evaluates the public framework without changing operator semantics.
"""

from .cascade import CascadePolicy, CascadeSignals, evaluate_cascade
from .critical_rupture import StructuralDefect, StructuralDiagnosis, diagnose_structural_ruptures
from .schemas import (
    ClaimStatus,
    EvidenceOrigin,
    ExternalGate,
    FidelityPair,
    HypothesisResult,
    OperationKind,
    PartitionRole,
    Q1CalibrationConfig,
    SplitUseRecord,
)
from .statistics import noninferiority_test
from .traceability import EvidenceClaim, MissingnessReport, evaluate_missingness, traceability_score

__all__ = [
    "CascadePolicy",
    "CascadeSignals",
    "ClaimStatus",
    "EvidenceClaim",
    "EvidenceOrigin",
    "ExternalGate",
    "FidelityPair",
    "HypothesisResult",
    "MissingnessReport",
    "OperationKind",
    "PartitionRole",
    "Q1CalibrationConfig",
    "SplitUseRecord",
    "StructuralDefect",
    "StructuralDiagnosis",
    "diagnose_structural_ruptures",
    "evaluate_cascade",
    "evaluate_missingness",
    "noninferiority_test",
    "traceability_score",
]
