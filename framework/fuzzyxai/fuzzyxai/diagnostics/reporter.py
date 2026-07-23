from __future__ import annotations

from dataclasses import asdict, replace

from .contracts import (
    DiagnosticCut,
    DiagnosticIssue,
    DiagnosticReport,
    RecertificationReport,
    RepairPlan,
    RouteGraph,
    ValidationResult,
    canonical_json,
    canonical_sha256,
)


class DiagnosticReporter:
    def build(
        self,
        graph: RouteGraph,
        validation: ValidationResult,
        issues: tuple[DiagnosticIssue, ...],
        cut: DiagnosticCut | None,
        repair_plan: RepairPlan | None,
        recertification: RecertificationReport | None,
    ) -> DiagnosticReport:
        limitations = [
            "Заключение относится к структурной целостности маршрута и не доказывает ошибочность прогноза.",
            "Ближайшая причина не считается корневой причинно-следственной связью без независимого журнала изменений.",
        ]
        if any(issue.insufficient_evidence for issue in issues):
            limitations.append("Часть обязательных сведений отсутствует; точная диагностика ограничена.")
        if cut and not cut.optimal:
            limitations.append("Оптимальность диагностического разреза не доказана.")
        if repair_plan and not repair_plan.fully_executable:
            limitations.append("План содержит рекомендации или шаги, требующие внешнего исполнения.")
        normalized_cut = replace(cut, runtime_ms=0.0) if cut else None
        normalized_plan = replace(repair_plan, cut=normalized_cut) if repair_plan and normalized_cut else repair_plan
        trace_payload = {
            "schema_version": "1.0",
            "route": graph.to_dict(),
            "validation": asdict(validation),
            "issues": [asdict(issue) for issue in issues],
            "minimal_cut": asdict(normalized_cut) if normalized_cut else None,
            "repair_plan": asdict(normalized_plan) if normalized_plan else None,
            "recertification": asdict(recertification) if recertification else None,
            "limitations": limitations,
        }
        trace = canonical_json(trace_payload)
        trace_hash = canonical_sha256(trace_payload)
        report_id = f"diagnostic:{graph.route_id}:{trace_hash[:12]}"
        return DiagnosticReport(
            report_id=report_id,
            route_id=graph.route_id,
            route_status=validation.status,
            issues=issues,
            minimal_cut=cut,
            repair_plan=repair_plan,
            recertification=recertification,
            user_summary=self._user_summary(validation, issues, recertification),
            expert_summary=self._expert_summary(validation, issues, cut, repair_plan, recertification),
            audit_summary=self._audit_summary(graph, validation, issues, cut, repair_plan, recertification),
            limitations=tuple(limitations),
            trace=trace,
            trace_sha256=trace_hash,
        )

    @staticmethod
    def _user_summary(
        validation: ValidationResult,
        issues: tuple[DiagnosticIssue, ...],
        recertification: RecertificationReport | None,
    ) -> str:
        if validation.valid:
            head = "Обязательные контракты объяснительного маршрута подтверждены."
        elif validation.status == "insufficient_evidence":
            head = "Объяснительный маршрут нельзя подтвердить: обязательных сведений недостаточно."
        else:
            head = f"Объяснительный маршрут не подтверждён; обнаружено нарушений: {len(issues)}."
        details = " ".join(issue.symptom for issue in issues[:3])
        repair = (
            f" Результат повторной проверки: {recertification.status}."
            if recertification
            else " Перед повторным использованием требуется выполнить предложенный план и заново проверить маршрут."
        )
        boundary = " Заключение относится к целостности маршрута и не означает, что прогноз модели обязательно ошибочен."
        return " ".join(part for part in (head, details, repair, boundary) if part).strip()

    @staticmethod
    def _expert_summary(
        validation: ValidationResult,
        issues: tuple[DiagnosticIssue, ...],
        cut: DiagnosticCut | None,
        repair_plan: RepairPlan | None,
        recertification: RecertificationReport | None,
    ) -> str:
        lines = [f"Статус: {validation.status}. Проверено контрактов: {len(validation.checked_contracts)}."]
        lines.extend(
            f"{issue.code}: {issue.violated_contract}; источник: {', '.join(issue.source_nodes) or 'не установлен'}."
            for issue in issues
        )
        if cut:
            lines.append(
                f"Диагностический разрез: {', '.join(cut.defect_atoms) or 'пуст'}; "
                f"стоимость {cut.total_cost:g}; optimal={cut.optimal}."
            )
        if repair_plan:
            lines.append(
                f"План {repair_plan.plan_id}: шагов {len(repair_plan.steps)}; "
                f"полностью исполним={repair_plan.fully_executable}."
            )
        if recertification:
            lines.append(f"Повторная проверка: {recertification.status}.")
        return "\n".join(lines)

    @staticmethod
    def _audit_summary(
        graph: RouteGraph,
        validation: ValidationResult,
        issues: tuple[DiagnosticIssue, ...],
        cut: DiagnosticCut | None,
        repair_plan: RepairPlan | None,
        recertification: RecertificationReport | None,
    ) -> str:
        return (
            f"route_id={graph.route_id}; schema={graph.schema_version}; nodes={len(graph.nodes)}; "
            f"edges={len(graph.edges)}; contracts={len(graph.contracts)}; "
            f"checked={len(validation.checked_contracts)}; issues={len(issues)}; "
            f"solver={cut.solver if cut else 'none'}; "
            f"plan={repair_plan.plan_id if repair_plan else 'none'}; "
            f"recertification={recertification.status if recertification else 'not_run'}; "
            f"route_sha256={graph.trace_sha256}."
        )
