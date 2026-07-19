"""Executable experiment entrypoints used by dissertation artifacts."""
from .contracts import (
    CalibrationConfig,
    CriticalRupture,
    CriticalRuptureType,
    DatasetSpec,
    ExperimentGate,
    ExperimentRunManifest,
    PolicyOutcome,
)
from .critical_rupture import (
    CRITICAL_RUPTURE_DEFINITION,
    detect_critical_ruptures,
    rupture_error_association,
)
from .statistics import PairedStatistic, holm_adjust, mcnemar_exact, paired_summary

__all__ = [
    "CRITICAL_RUPTURE_DEFINITION",
    "CalibrationConfig",
    "CriticalRupture",
    "CriticalRuptureType",
    "DatasetSpec",
    "ExperimentGate",
    "ExperimentRunManifest",
    "PairedStatistic",
    "PolicyOutcome",
    "detect_critical_ruptures",
    "holm_adjust",
    "mcnemar_exact",
    "paired_summary",
    "rupture_error_association",
]
