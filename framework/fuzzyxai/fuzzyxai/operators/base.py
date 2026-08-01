from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class OperatorContract:
    """Machine-checkable boundary between two diagnostic operators."""

    contract_id: str
    required_fields: frozenset[str] = frozenset()
    produced_fields: frozenset[str] = frozenset()
    constraints: Mapping[str, object] = field(default_factory=dict)

    def compatible_with(self, downstream: OperatorContract) -> bool:
        if not downstream.required_fields.issubset(self.produced_fields):
            return False
        shared = set(self.constraints).intersection(downstream.constraints)
        return all(self.constraints[key] == downstream.constraints[key] for key in shared)


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    checks: tuple[Mapping[str, object], ...] = ()
    code: str = "PASS"
    message: str = ""


@dataclass(frozen=True)
class RouteState:
    values: Mapping[str, object]
    evidence_refs: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    critical_violations: tuple[str, ...] = ()
    operator_trace: tuple[str, ...] = ()

    def updated(self, operator_id: str, **values: object) -> RouteState:
        merged = dict(self.values)
        merged.update(values)
        return replace(self, values=merged, operator_trace=(*self.operator_trace, operator_id))


@runtime_checkable
class DiagnosticOperator(Protocol):
    operator_id: str
    input_contract: OperatorContract
    output_contract: OperatorContract

    def apply(self, state: RouteState) -> RouteState: ...

    def verify_preconditions(self, state: RouteState) -> VerificationResult: ...

    def verify_postconditions(self, state: RouteState) -> VerificationResult: ...


class OperatorExecutionError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        operator_id: str | None = None,
        checks: tuple[Mapping[str, object], ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.operator_id = operator_id
        self.checks = checks

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": str(self),
            "operator_id": self.operator_id,
            "checks": [dict(check) for check in self.checks],
        }


@dataclass(frozen=True)
class FunctionalDiagnosticOperator:
    operator_id: str
    input_contract: OperatorContract
    output_contract: OperatorContract
    function: Callable[[RouteState], RouteState]

    def verify_preconditions(self, state: RouteState) -> VerificationResult:
        missing = tuple(sorted(self.input_contract.required_fields - set(state.values)))
        checks = tuple(
            {"check": f"required_field:{field}", "passed": field not in missing}
            for field in sorted(self.input_contract.required_fields)
        )
        return VerificationResult(
            passed=not missing,
            checks=checks,
            code="PASS" if not missing else "OPERATOR_PRECONDITION_FAILED",
            message="" if not missing else f"missing required fields: {', '.join(missing)}",
        )

    def apply(self, state: RouteState) -> RouteState:
        preconditions = self.verify_preconditions(state)
        if not preconditions.passed:
            raise OperatorExecutionError(
                preconditions.code,
                preconditions.message,
                operator_id=self.operator_id,
                checks=preconditions.checks,
            )
        try:
            result = self.function(state)
        except OperatorExecutionError:
            raise
        except Exception as exc:
            raise OperatorExecutionError(
                "OPERATOR_EXECUTION_FAILED",
                str(exc),
                operator_id=self.operator_id,
            ) from exc
        postconditions = self.verify_postconditions(result)
        if not postconditions.passed:
            raise OperatorExecutionError(
                postconditions.code,
                postconditions.message,
                operator_id=self.operator_id,
                checks=postconditions.checks,
            )
        return result

    def verify_postconditions(self, state: RouteState) -> VerificationResult:
        missing = tuple(sorted(self.output_contract.produced_fields - set(state.values)))
        checks = tuple(
            {"check": f"produced_field:{field}", "passed": field not in missing}
            for field in sorted(self.output_contract.produced_fields)
        )
        return VerificationResult(
            passed=not missing and not state.critical_violations,
            checks=checks,
            code=(
                "PASS"
                if not missing and not state.critical_violations
                else "OPERATOR_POSTCONDITION_FAILED"
            ),
            message=(
                ""
                if not missing and not state.critical_violations
                else "output contract or critical-violation check failed"
            ),
        )
