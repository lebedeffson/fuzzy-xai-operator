from __future__ import annotations

from collections import Counter
from time import perf_counter
from typing import Literal

from .contracts import (
    BatchDiagnosticReport,
    DiagnosticReport,
    RepairCostModel,
    RepairExecutionContext,
)
from .minimal_cut import MinimalDiagnosticCutFinder
from .recertification import RouteRecertifier
from .repair_executor import RepairExecutor
from .repair_planner import ActionableRepairPlanner
from .reporter import DiagnosticReporter
from .route_graph import RouteGraphBuilder
from .validator import DiagnosticValidator


class DiagnosticService:
    def __init__(
        self,
        *,
        builder: RouteGraphBuilder | None = None,
        validator: DiagnosticValidator | None = None,
        cut_finder: MinimalDiagnosticCutFinder | None = None,
        repair_planner: ActionableRepairPlanner | None = None,
        repair_executor: RepairExecutor | None = None,
        recertifier: RouteRecertifier | None = None,
        reporter: DiagnosticReporter | None = None,
    ) -> None:
        self.builder = builder or RouteGraphBuilder()
        self.validator = validator or DiagnosticValidator()
        self.cut_finder = cut_finder or MinimalDiagnosticCutFinder()
        self.repair_planner = repair_planner or ActionableRepairPlanner()
        self.repair_executor = repair_executor or RepairExecutor(self.repair_planner.registry)
        self.recertifier = recertifier or RouteRecertifier(self.validator)
        self.reporter = reporter or DiagnosticReporter()

    def diagnose(
        self,
        *,
        route: object,
        repair_mode: Literal["none", "plan", "execute"] = "plan",
        repair_context: RepairExecutionContext | None = None,
        audience: Literal["user", "expert", "audit"] = "user",
    ) -> DiagnosticReport:
        del audience  # all deterministic levels are always retained in the report
        if repair_mode not in {"none", "plan", "execute"}:
            raise ValueError("repair_mode must be none, plan, or execute")
        graph = self.builder.build(route)
        validation = self.validator.validate(graph)
        cut = None
        plan = None
        recertification = None
        if validation.issues and repair_mode != "none":
            costs = RepairCostModel(graph.metadata.get("repair_costs", {}))
            cut = self.cut_finder.find(graph, validation, costs)
            plan = self.repair_planner.plan(graph, validation.issues, cut)
        if repair_mode == "execute":
            if repair_context is None:
                raise PermissionError("repair_mode='execute' requires an explicit RepairExecutionContext")
            if plan is None:
                raise ValueError("no repair plan is available")
            repaired, execution_results = self.repair_executor.execute(graph, plan, repair_context)
            recertification = self.recertifier.recertify(graph, repaired, plan, execution_results)
        return self.reporter.build(graph, validation, validation.issues, cut, plan, recertification)

    def diagnose_batch(
        self,
        *,
        routes: object,
        repair_mode: Literal["none", "plan"] = "plan",
    ) -> BatchDiagnosticReport:
        if repair_mode == "execute":
            raise ValueError("batch execution is disabled; execute explicitly per route")
        started = perf_counter()
        reports = tuple(self.diagnose(route=route, repair_mode=repair_mode) for route in routes)
        statuses = Counter(report.route_status for report in reports)
        categories = Counter(issue.category for report in reports for issue in report.issues)
        sources = Counter(node for report in reports for issue in report.issues for node in issue.source_nodes)
        cuts = Counter(report.minimal_cut.defect_atoms for report in reports if report.minimal_cut)
        plans = [report.repair_plan for report in reports if report.repair_plan]
        executable = sum(plan.fully_executable for plan in plans) / len(plans) if plans else 0.0
        return BatchDiagnosticReport(
            reports=reports,
            route_status_counts=dict(sorted(statuses.items())),
            issue_category_counts=dict(sorted(categories.items())),
            frequent_source_nodes=tuple(sources.most_common()),
            frequent_cuts=tuple(cuts.most_common()),
            fully_executable_plan_rate=executable,
            unknown_issue_count=sum(issue.unknown for report in reports for issue in report.issues),
            runtime_ms=(perf_counter() - started) * 1000,
        )


def diagnose_route(
    route: object,
    *,
    repair_mode: Literal["none", "plan", "execute"] = "plan",
    repair_context: RepairExecutionContext | None = None,
) -> DiagnosticReport:
    return DiagnosticService().diagnose(
        route=route,
        repair_mode=repair_mode,
        repair_context=repair_context,
    )
