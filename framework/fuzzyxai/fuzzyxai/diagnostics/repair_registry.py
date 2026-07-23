from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .contracts import DiagnosticIssue, RepairStep, RouteGraph


@dataclass(frozen=True)
class PreconditionResult:
    satisfied: bool
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class StepVerification:
    passed: bool
    checks: tuple[dict[str, object], ...] = ()


class RepairProvider(Protocol):
    provider_id: str

    def supports(self, issue: DiagnosticIssue) -> bool: ...

    def check_preconditions(self, graph: RouteGraph, issue: DiagnosticIssue) -> PreconditionResult: ...

    def propose(self, graph: RouteGraph, issue: DiagnosticIssue) -> tuple[RepairStep, ...]: ...

    def verify(self, before: RouteGraph, after: RouteGraph, step: RepairStep) -> StepVerification: ...


@dataclass(frozen=True)
class ContractRepairProvider:
    provider_id: str
    codes: frozenset[str]
    categories: frozenset[str]
    operation: str
    title: str
    cost: float
    automatic: bool = False

    def supports(self, issue: DiagnosticIssue) -> bool:
        return issue.code in self.codes or issue.category in self.categories

    def check_preconditions(self, graph: RouteGraph, issue: DiagnosticIssue) -> PreconditionResult:
        del graph
        requirements = tuple(f"registered_source_available:{node}" for node in issue.source_nodes)
        # Availability must be resolved by the execution context. A plan can be
        # proposed without pretending that an external artifact exists.
        return PreconditionResult(True, requirements)

    def propose(self, graph: RouteGraph, issue: DiagnosticIssue) -> tuple[RepairStep, ...]:
        del graph
        targets = issue.source_nodes or issue.affected_edges or issue.affected_nodes
        return tuple(
            RepairStep(
                step_id=f"step:{issue.issue_id}:{index}",
                title=self.title,
                target=target,
                provider_id=self.provider_id,
                operation=self.operation,
                parameters={
                    "issue_id": issue.issue_id,
                    "contract_id": issue.violated_contract,
                    "registered_source_ref": f"registry://{target}",
                },
                preconditions=(f"registered_source_available:{target}",),
                depends_on=(),
                expected_postconditions=(f"contract_satisfied:{issue.violated_contract}",),
                verification_checks=(
                    f"revalidate:{issue.violated_contract}",
                    "rebuild_route",
                    "recertify_route",
                ),
                fallback_step_ids=(),
                rollback_operation="restore_previous_artifact_snapshot",
                estimated_cost=self.cost,
                requires_human_approval=not self.automatic,
                executable=self.automatic,
            )
            for index, target in enumerate(targets, 1)
        )

    def verify(self, before: RouteGraph, after: RouteGraph, step: RepairStep) -> StepVerification:
        changed = before.trace_sha256 != after.trace_sha256
        return StepVerification(
            passed=changed,
            checks=(
                {
                    "check": "route_changed",
                    "passed": changed,
                    "step_id": step.step_id,
                },
            ),
        )


def _default_providers() -> tuple[RepairProvider, ...]:
    return (
        ContractRepairProvider(
            "provenance.restore",
            frozenset({"insufficient_evidence"}),
            frozenset({"provenance"}),
            "restore_registered_provenance",
            "Восстановить зарегистрированное происхождение",
            1.0,
        ),
        ContractRepairProvider(
            "model.reload",
            frozenset({"contract_mismatch"}),
            frozenset({"model"}),
            "load_registered_model_version",
            "Загрузить зарегистрированную версию модели",
            8.0,
        ),
        ContractRepairProvider(
            "preprocessing.restore",
            frozenset({"contract_mismatch"}),
            frozenset({"preprocessing", "data"}),
            "restore_registered_preprocessing",
            "Восстановить зарегистрированную предобработку",
            2.0,
        ),
        ContractRepairProvider(
            "calibration.refresh",
            frozenset({"contract_max_value_failed", "contract_mismatch"}),
            frozenset({"calibration"}),
            "refresh_calibration_artifact",
            "Получить действующий калибровочный артефакт",
            2.0,
        ),
        ContractRepairProvider(
            "artifact.restore",
            frozenset({"checksum_mismatch"}),
            frozenset({"serialization"}),
            "restore_artifact_by_verified_hash",
            "Восстановить артефакт с проверяемой контрольной суммой",
            2.0,
        ),
        ContractRepairProvider(
            "field.restore",
            frozenset({"contract_required_attribute_failed", "insufficient_evidence"}),
            frozenset({"runtime"}),
            "obtain_required_field_from_registered_source",
            "Получить отсутствующее обязательное поле",
            1.0,
        ),
        ContractRepairProvider(
            "dependency.restore",
            frozenset({"contract_edge_present_failed"}),
            frozenset({"provenance"}),
            "restore_registered_dependency",
            "Восстановить обязательную зависимость",
            1.0,
        ),
        ContractRepairProvider(
            "reduction.rebuild",
            frozenset({"contract_max_value_failed", "contract_mismatch"}),
            frozenset({"reduction"}),
            "rebuild_reduction_with_registered_ceiling",
            "Перестроить редукцию в зарегистрированной границе потерь",
            4.0,
        ),
        ContractRepairProvider(
            "dictionary.restore",
            frozenset({"contract_mismatch"}),
            frozenset({"representation"}),
            "load_registered_dictionary",
            "Загрузить совместимую версию словаря",
            2.0,
        ),
        ContractRepairProvider(
            "explainer.rerun",
            frozenset({"contract_mismatch", "insufficient_evidence"}),
            frozenset({"explainer"}),
            "rerun_explainer_with_registered_components",
            "Повторно построить объяснительный артефакт",
            4.0,
        ),
    )


@dataclass
class RepairProviderRegistry:
    providers: list[RepairProvider] = field(default_factory=lambda: list(_default_providers()))

    def register(self, provider: RepairProvider) -> None:
        if any(item.provider_id == provider.provider_id for item in self.providers):
            raise ValueError(f"repair provider already registered: {provider.provider_id}")
        self.providers.append(provider)

    def select(self, issue: DiagnosticIssue) -> RepairProvider | None:
        return next((provider for provider in self.providers if provider.supports(issue)), None)

    def get(self, provider_id: str) -> RepairProvider:
        try:
            return next(provider for provider in self.providers if provider.provider_id == provider_id)
        except StopIteration as exc:
            raise KeyError(provider_id) from exc
