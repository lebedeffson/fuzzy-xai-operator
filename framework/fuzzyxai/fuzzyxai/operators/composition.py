from __future__ import annotations

from dataclasses import dataclass

from .base import (
    DiagnosticOperator,
    OperatorExecutionError,
    RouteState,
    VerificationResult,
)


@dataclass(frozen=True)
class ExplanationArtifact:
    artifact_id: str
    method: str
    model_version: str
    preprocessing_version: str
    sample_id: str
    provenance_ref: str | None


class ComposedDiagnosticOperator:
    """Sequential, fail-closed composition of two contract operators."""

    def __init__(
        self,
        first: DiagnosticOperator,
        second: DiagnosticOperator,
    ) -> None:
        if not first.output_contract.compatible_with(second.input_contract):
            raise OperatorExecutionError(
                "OPERATOR_CONTRACT_MISMATCH",
                (
                    f"{first.operator_id} output {first.output_contract.contract_id} "
                    f"is incompatible with {second.operator_id} input "
                    f"{second.input_contract.contract_id}"
                ),
                operator_id=f"{first.operator_id}->{second.operator_id}",
            )
        self.first = first
        self.second = second
        self.operator_id = f"{first.operator_id}|{second.operator_id}"
        self.input_contract = first.input_contract
        self.output_contract = second.output_contract

    @staticmethod
    def _require(
        result: VerificationResult,
        *,
        operator_id: str,
        phase: str,
    ) -> None:
        if result.passed:
            return
        raise OperatorExecutionError(
            result.code,
            f"{operator_id} {phase}: {result.message}",
            operator_id=operator_id,
            checks=result.checks,
        )

    def verify_preconditions(self, state: RouteState) -> VerificationResult:
        return self.first.verify_preconditions(state)

    def apply(self, state: RouteState) -> RouteState:
        self._require(
            self.first.verify_preconditions(state),
            operator_id=self.first.operator_id,
            phase="precondition",
        )
        try:
            intermediate = self.first.apply(state)
        except OperatorExecutionError:
            raise
        except Exception as exc:
            raise OperatorExecutionError(
                "OPERATOR_EXECUTION_FAILED",
                str(exc),
                operator_id=self.first.operator_id,
            ) from exc
        self._require(
            self.first.verify_postconditions(intermediate),
            operator_id=self.first.operator_id,
            phase="postcondition",
        )
        self._require(
            self.second.verify_preconditions(intermediate),
            operator_id=self.second.operator_id,
            phase="precondition",
        )
        try:
            result = self.second.apply(intermediate)
        except OperatorExecutionError:
            raise
        except Exception as exc:
            raise OperatorExecutionError(
                "OPERATOR_EXECUTION_FAILED",
                str(exc),
                operator_id=self.second.operator_id,
            ) from exc
        self._require(
            self.second.verify_postconditions(result),
            operator_id=self.second.operator_id,
            phase="postcondition",
        )
        return result

    def verify_postconditions(self, state: RouteState) -> VerificationResult:
        return self.second.verify_postconditions(state)


def compose_operators(*operators: DiagnosticOperator) -> DiagnosticOperator:
    if not operators:
        raise ValueError("at least one operator is required")
    composed: DiagnosticOperator = operators[0]
    for operator in operators[1:]:
        composed = ComposedDiagnosticOperator(composed, operator)
    return composed


def route_explanation_artifacts(
    shap: ExplanationArtifact,
    lime: ExplanationArtifact,
) -> RouteState:
    """Create a shared diagnostic state without combining attribution values."""

    checks = {
        "same_model_version": shap.model_version == lime.model_version,
        "same_preprocessing_version": (
            shap.preprocessing_version == lime.preprocessing_version
        ),
        "same_sample_id": shap.sample_id == lime.sample_id,
        "shap_provenance_registered": bool(shap.provenance_ref),
        "lime_provenance_registered": bool(lime.provenance_ref),
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    if failures:
        raise OperatorExecutionError(
            "OPERATOR_CONTRACT_MISMATCH",
            f"SHAP/LIME route compatibility failed: {', '.join(failures)}",
            operator_id="compose_shap_lime_route",
            checks=tuple(
                {"check": name, "passed": passed}
                for name, passed in checks.items()
            ),
        )
    return RouteState(
        values={
            "shap_artifact": shap,
            "lime_artifact": lime,
            "model_version": shap.model_version,
            "preprocessing_version": shap.preprocessing_version,
            "sample_id": shap.sample_id,
            "diagnostic_route_kind": "parallel_explanation_artifacts",
        },
        evidence_refs=(str(shap.provenance_ref), str(lime.provenance_ref)),
        operator_trace=("compose_shap_lime_route",),
    )
