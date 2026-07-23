from __future__ import annotations

import warnings

from fuzzyxai.audit_h10.models import AuditDiagnosis, RouteObservation

from .contracts import DiagnosticReport
from .service import DiagnosticService


LEGACY_POLICY_WARNING = (
    "This mode is retained only to reproduce earlier experiments. "
    "Its advantage over strong threshold policies was not supported."
)


def diagnose_h10_observation(route: RouteObservation) -> DiagnosticReport:
    """Route an old H10 observation through the v21 diagnostic service."""

    warnings.warn(LEGACY_POLICY_WARNING, DeprecationWarning, stacklevel=2)
    return DiagnosticService().diagnose(route=route)


def legacy_audit_boundary(_: AuditDiagnosis) -> str:
    """Return the claim boundary without converting old frozen evidence."""

    return LEGACY_POLICY_WARNING
