from __future__ import annotations

from .contracts import (
    RepairExecutionContext,
    RepairPlan,
    RouteGraph,
    StepExecutionResult,
)
from .repair_registry import RepairProviderRegistry


class RepairExecutor:
    """Execute only explicitly approved registered handlers.

    This production component has no access to benchmark mutation logs, source
    truth, repair truth, or hidden benchmark targets.
    """

    def __init__(self, registry: RepairProviderRegistry | None = None) -> None:
        self.registry = registry or RepairProviderRegistry()

    def execute(
        self,
        graph: RouteGraph,
        plan: RepairPlan,
        context: RepairExecutionContext,
    ) -> tuple[RouteGraph, tuple[StepExecutionResult, ...]]:
        if not context.allow_external_changes:
            raise PermissionError("external repair execution requires allow_external_changes=True")
        current = graph
        results: list[StepExecutionResult] = []
        completed: set[str] = set()
        for step in plan.steps:
            if step.requires_human_approval and step.step_id not in context.approved_step_ids:
                results.append(StepExecutionResult(step.step_id, "not_approved", False))
                continue
            if any(dependency not in completed for dependency in step.depends_on):
                results.append(StepExecutionResult(step.step_id, "dependency_failed", False))
                continue
            handler = context.handlers.get(step.operation)
            if handler is None:
                results.append(StepExecutionResult(step.step_id, "handler_unavailable", False))
                continue
            missing_preconditions = tuple(
                condition
                for condition in step.preconditions
                if condition not in context.satisfied_preconditions
                and not condition.startswith("registered_source_available:")
            )
            if missing_preconditions:
                results.append(
                    StepExecutionResult(
                        step.step_id,
                        "precondition_failed",
                        False,
                        verification=tuple(
                            {
                                "check": condition,
                                "passed": False,
                            }
                            for condition in missing_preconditions
                        ),
                    )
                )
                continue
            before = current
            try:
                candidate = handler(current, step)
                if not isinstance(candidate, RouteGraph):
                    raise TypeError("repair handler must return RouteGraph")
                verification = self.registry.get(step.provider_id).verify(before, candidate, step)
                status = "completed" if verification.passed else "verification_failed"
                rollback_verified = None
                if verification.passed:
                    current = candidate
                else:
                    rollback_handler = (
                        context.handlers.get(step.rollback_operation)
                        if step.rollback_operation
                        else None
                    )
                    rolled_back = rollback_handler(candidate, step) if rollback_handler else before
                    if not isinstance(rolled_back, RouteGraph):
                        raise TypeError("rollback handler must return RouteGraph")
                    rollback_verified = rolled_back.trace_sha256 == before.trace_sha256
                    if not rollback_verified:
                        raise RuntimeError("repair rollback checksum mismatch")
                    current = rolled_back
                results.append(
                    StepExecutionResult(
                        step.step_id,
                        status,
                        before.trace_sha256 != candidate.trace_sha256,
                        verification.checks,
                        rollback_verified=rollback_verified,
                    )
                )
                if verification.passed:
                    completed.add(step.step_id)
            except Exception as exc:  # external providers define their own failures
                current = before
                results.append(
                    StepExecutionResult(
                        step.step_id,
                        "failed",
                        False,
                        error=str(exc),
                        rollback_verified=current.trace_sha256 == before.trace_sha256,
                    )
                )
        return current, tuple(results)
