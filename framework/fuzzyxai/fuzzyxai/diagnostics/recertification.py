from __future__ import annotations

from .contracts import (
    RecertificationReport,
    RepairPlan,
    RouteGraph,
    StepExecutionResult,
)
from .validator import DiagnosticValidator


class RouteRecertifier:
    def __init__(self, validator: DiagnosticValidator | None = None) -> None:
        self.validator = validator or DiagnosticValidator()

    def recertify(
        self,
        before: RouteGraph,
        after: RouteGraph,
        plan: RepairPlan,
        execution_results: tuple[StepExecutionResult, ...],
    ) -> RecertificationReport:
        before_result = self.validator.validate(before)
        after_result = self.validator.validate(after)
        before_codes = {issue.issue_id for issue in before_result.issues}
        after_codes = {issue.issue_id for issue in after_result.issues}
        completed = tuple(result.step_id for result in execution_results if result.status == "completed")
        failed = tuple(result.step_id for result in execution_results if result.status != "completed")
        if after_result.valid and not failed:
            status = "complete_recovery"
        elif after_result.valid:
            status = "complete_recovery_with_skipped_steps"
        elif len(after_codes) < len(before_codes):
            status = "partial_recovery"
        elif after_codes - before_codes:
            status = "degraded"
        elif before.trace_sha256 == after.trace_sha256:
            status = "no_change"
        else:
            status = "verification_inconclusive"
        checks = tuple(
            {
                "step_id": result.step_id,
                "status": result.status,
                "changed": result.changed,
                "verification": result.verification,
            }
            for result in execution_results
        )
        return RecertificationReport(
            status=status,
            route_valid_before=before_result.valid,
            route_valid_after=after_result.valid,
            completed_steps=completed,
            failed_steps=failed,
            resolved_issues=tuple(sorted(before_codes - after_codes)),
            remaining_issues=tuple(sorted(before_codes & after_codes)),
            new_issues=tuple(sorted(after_codes - before_codes)),
            verification_results=checks,
            before_trace_sha256=before.trace_sha256,
            after_trace_sha256=after.trace_sha256,
        )
