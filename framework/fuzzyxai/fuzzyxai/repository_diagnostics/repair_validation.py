from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepairExecutionEvidence:
    bug_reproduced: bool
    plan_executed: bool
    fail_to_pass: bool
    regression_passed: bool
    recertification_passed: bool
    new_critical_violations: int
    rollback_passed: bool


def classify_repair_execution(
    evidence: RepairExecutionEvidence,
) -> str:
    if not evidence.bug_reproduced:
        return "BUG_NOT_REPRODUCED"
    if not evidence.plan_executed:
        return "PLAN_EXECUTION_FAILED"
    if not evidence.fail_to_pass:
        return "FAIL_TO_PASS_FAILED"
    if not evidence.regression_passed:
        return "REGRESSION_FAILED"
    if (
        not evidence.recertification_passed
        or evidence.new_critical_violations
    ):
        return "RECERTIFICATION_FAILED"
    if not evidence.rollback_passed:
        return "ROLLBACK_FAILED"
    return "RESTORATION_CONFIRMED"
