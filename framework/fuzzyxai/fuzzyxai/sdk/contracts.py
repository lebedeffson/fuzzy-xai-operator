from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

AdapterStatus = Literal['ok', 'warning', 'error']
ArtifactType = Literal['ExplanationArtifact', 'DiagnosticArtifact', 'ReportArtifact']


@dataclass(frozen=True)
class AdapterMetadata:
    registry_id: str
    adapter_class: str
    input_types: list[str]
    output_type: ArtifactType = 'ExplanationArtifact'
    claim_scope: str = 'adapter route only, not model validation'
    version: str = '1.0'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    ok: bool
    status: AdapterStatus
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExplanationArtifact:
    registry_id: str
    artifact_type: ArtifactType
    has_explanation_object: bool
    has_diagnostic_state: bool
    payload: dict[str, Any]
    trace: dict[str, Any]
    report_path: str = ''

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RiskActionResult:
    registry_id: str
    chi_R: int | str
    chi_Auto: bool | str
    rho: float | str
    action: str
    status: str
    notes: str = ''

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
