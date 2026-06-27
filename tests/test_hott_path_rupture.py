from __future__ import annotations

from fuzzyxai.core.diagnostics import DiagnosticState
from fuzzyxai.hott import RuptureType, certify_rupture


def test_hott_rupture_certificate_smoke() -> None:
    diagnostic = DiagnosticState('D_ij', 'semantic disagreement exceeds gamma_max', 'critical', {'gamma': 0.7, 'gamma_max': 0.45})
    rupture = RuptureType.from_diagnostic('E_model', 'E_risk', diagnostic, gamma=0.7, threshold=0.45)
    cert = certify_rupture(rupture)
    assert cert.diagnostic_code == 'D_ij'
    assert cert.gamma == 0.7
    assert cert.gamma_max == 0.45
