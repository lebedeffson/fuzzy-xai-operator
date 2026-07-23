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
            if step.step_id not in context.approved_step_ids:
                results.append(StepExecutionResult(step.step_id, "not_approved", False))
                continue
            if any(dependency not in completed for dependency in step.depends_on):
                results.append(StepExecutionResult(step.step_id, "dependency_failed", False))
                continue
            handler = context.handlers.get(step.operation)
            if handler is None:
                results.append(StepExecutionResult(step.step_id, "handler_unavailable", False))
                continue
            before = current
            try:
                current = handler(current, step)
                if not isinstance(current, RouteGraph):
                    raise TypeError("repair handler must return RouteGraph")
                verification = self.registry.get(step.provider_id).verify(before, current, step)
                status = "completed" if verification.passed else "verification_failed"
                results.append(
                    StepExecutionResult(
                        step.step_id,
                        status,
                        before.trace_sha256 != current.trace_sha256,
                        verification.checks,
                    )
                )
                if verification.passed:
                    completed.add(step.step_id)
            except Exception as exc:  # external providers define their own failures
                current = before
                results.append(StepExecutionResult(step.step_id, "failed", False, error=str(exc)))
        return current, tuple(results)
