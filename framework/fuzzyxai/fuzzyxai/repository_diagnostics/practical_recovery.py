from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .auditor import AuditCandidate
from .recovery import (
    IncidentExecutionReport,
    IncidentSandboxExecutor,
    RegisteredRepair,
)

AutomationLevel = Literal["AUTOMATIC", "HUMAN_APPROVAL", "LOCALIZATION_ONLY"]


@dataclass(frozen=True)
class VerifiableRepairPlan:
    automation_level: AutomationLevel
    operation: str
    target_file: str | None
    target_symbol: str | None
    preconditions: tuple[str, ...]
    patch_preview: str
    affected_files: tuple[str, ...]
    rollback: str
    fail_to_pass_command: tuple[str, ...]
    regression_command: tuple[str, ...]
    executable: bool
    requires_human_approval: bool


class PracticalRepairPlanner:
    """Map supported contracts to bounded repairs, otherwise localize only."""

    def plan(
        self,
        candidate: AuditCandidate,
        failing_tests: tuple[str, ...],
    ) -> VerifiableRepairPlan:
        automatic = {
            "ARTIFACT_PROVENANCE": "restore_registered_artifact_digest",
            "DEPENDENCY_VERSION": "pin_registered_dependency_version",
            "MODEL_EXPLAINER_VERSION": "align_registered_model_explainer_versions",
        }
        approval = {
            "DATA_CONTRACT": "update_registered_input_adapter",
            "MODEL_LOADING": "update_registered_checkpoint_loader",
            "PIPELINE_CONFIGURATION": "replace_registered_configuration_value",
            "SERIALIZATION": "align_registered_reader_writer_options",
        }
        if candidate.contract in automatic:
            level: AutomationLevel = "AUTOMATIC"
            operation = automatic[candidate.contract]
        elif candidate.contract in approval:
            level = "HUMAN_APPROVAL"
            operation = approval[candidate.contract]
        else:
            level = "LOCALIZATION_ONLY"
            operation = "no_registered_repair"
        target = candidate.file_path
        command = ("python", "-m", "pytest", failing_tests[0], "-x", "-vv") if failing_tests else ()
        executable = (
            level != "LOCALIZATION_ONLY"
            and bool(target)
            and bool(failing_tests)
        )
        return VerifiableRepairPlan(
            level,
            operation,
            target,
            candidate.symbol,
            (
                "target exists in buggy snapshot",
                "registered expected value is available",
                "sandbox is isolated",
            ),
            (f"{operation} at {target}:{candidate.symbol}" if executable else "No executable patch is registered"),
            (target,) if target else (),
            "restore sandbox snapshot",
            command,
            command,
            executable,
            level == "HUMAN_APPROVAL",
        )


@dataclass(frozen=True)
class RegisteredRepairProvider:
    operation: str
    apply: Callable[[Path], None]


class PracticalRepairExecutor:
    """Execute a plan only through an explicitly registered sandbox provider."""

    def __init__(
        self,
        providers: tuple[RegisteredRepairProvider, ...],
    ) -> None:
        self.providers = {provider.operation: provider for provider in providers}
        self.sandbox = IncidentSandboxExecutor()

    def execute(
        self,
        plan: VerifiableRepairPlan,
        source_root: Path,
        *,
        approved: bool,
        recertify: Callable[[Path], bool],
        timeout_seconds: int = 300,
    ) -> IncidentExecutionReport:
        if not plan.executable:
            raise ValueError("repair plan is localization-only")
        if plan.requires_human_approval and not approved:
            raise PermissionError("repair plan requires human approval")
        provider = self.providers.get(plan.operation)
        if provider is None or not callable(provider.apply):
            raise ValueError(f"registered repair provider is missing: {plan.operation}")
        repair = RegisteredRepair(
            f"h10-c5c:{plan.operation}",
            plan.operation,
            plan.target_file or "",
            provider.apply,
        )
        return self.sandbox.execute(
            source_root,
            repair,
            fail_to_pass_command=plan.fail_to_pass_command,
            regression_command=plan.regression_command,
            recertify=recertify,
            timeout_seconds=timeout_seconds,
        )
