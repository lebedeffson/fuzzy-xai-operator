from typing import Set
from fuzzyxai.core.diagnostics import DiagnosticState

def choice_diagnostic(profile: Set[str], covered: Set[str]) -> DiagnosticState:
    return DiagnosticState('D_choice', 'profile is not covered by available classes', 'critical', {
        'missing': sorted(profile - covered),
        'action': 'extend hierarchy or relax profile'
    })
