from dataclasses import dataclass, field
from typing import Any, Dict

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
