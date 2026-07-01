from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class DiagnosticType(Enum):
    INCOMPLETE_INTERFACE = "D_k_incomplete"
    TRACE_MISSING = "D_k_trace"
    NORMALIZATION_FAILED = "D_k_norm"
    RULE_CONFLICT = "D_k_rule"
    INTERFACE_RUPTURE = "D_ij"
    UNCERTAINTY_CHOICE_FAILED = "D_choice"
    QUALITY_SOURCE_CONFLICT = "D_quality_source_conflict"
    RULE_ATTRIBUTION_CONFLICT = "D_rule_attribution_conflict"
    COUNTEREVIDENCE_CONFLICT = "D_counterevidence_conflict"

@dataclass(frozen=True)
class DiagnosticState:
    """Diagnostic object D_ij or D_choice.

    It is returned when an explanation cannot be constructed, aligned, composed,
    or when a situation profile is not covered by the available hierarchy.
    """
    code: str
    reason: str
    severity: str = 'warning'
    context: Dict[str, Any] = field(default_factory=dict)

    def is_critical(self) -> bool:
        return self.severity.lower() in {'critical', 'error', 'fatal'}
