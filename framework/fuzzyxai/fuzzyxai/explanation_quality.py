from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ExplanationQualityReport:
    faithfulness: float | None
    fidelity: float | None
    stability: float | None
    completeness: float | None
    sparsity: float | None
    reconstruction_error: float | None
    status: str
    blocked_channels: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_quality_report(metrics: Mapping[str, Any], *, regression: bool = False) -> ExplanationQualityReport:
    fidelity = _optional_float(metrics.get("fidelity"))
    stability = _optional_float(metrics.get("stability"))
    completeness = _optional_float(metrics.get("trace_completeness", metrics.get("completeness")))
    sparsity = _optional_float(metrics.get("sparsity"))
    reconstruction_error = _optional_float(metrics.get("reconstruction_error"))
    faithfulness = _optional_float(metrics.get("faithfulness"))
    threshold = 0.80 if regression else 0.90
    blocked: list[str] = []
    limitations: list[str] = []
    if fidelity is not None and fidelity < threshold:
        blocked.append("surrogate_local_contributions")
        limitations.append(f"Surrogate fidelity {fidelity:.3f} is below the required {threshold:.2f}.")
    if reconstruction_error is not None and reconstruction_error > 1e-5:
        blocked.append("native_additive_contributions")
        limitations.append("Native additive contributions do not reconstruct the documented model output.")
    if completeness is None:
        limitations.append("Trace completeness was not measured.")
    status = "insufficient_evidence" if blocked else "pass" if completeness is not None else "partial"
    return ExplanationQualityReport(
        faithfulness=faithfulness,
        fidelity=fidelity,
        stability=stability,
        completeness=completeness,
        sparsity=sparsity,
        reconstruction_error=reconstruction_error,
        status=status,
        blocked_channels=tuple(blocked),
        limitations=tuple(limitations),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
