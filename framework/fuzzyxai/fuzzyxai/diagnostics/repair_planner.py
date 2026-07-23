from __future__ import annotations

from dataclasses import replace

from .contracts import DiagnosticCut, DiagnosticIssue, RepairPlan, RouteGraph, canonical_sha256
from .repair_registry import RepairProviderRegistry


class ActionableRepairPlanner:
    def __init__(self, registry: RepairProviderRegistry | None = None) -> None:
        self.registry = registry or RepairProviderRegistry()

    def plan(
        self,
        graph: RouteGraph,
        issues: tuple[DiagnosticIssue, ...],
        cut: DiagnosticCut,
    ) -> RepairPlan:
        selected_issues = tuple(
            issue
            for issue in issues
            if any(
                any(
                    f"node:{subject}/" in atom or f"edge:{subject}/" in atom
                    for subject in (*issue.source_nodes, *issue.affected_nodes, *issue.affected_edges)
                )
                for atom in cut.defect_atoms
            )
        )
        steps = []
        unresolved: list[str] = list(cut.uncovered_obligations)
        for issue in selected_issues:
            provider = self.registry.select(issue)
            if provider is None:
                unresolved.append(issue.issue_id)
                continue
            proposed = provider.propose(graph, issue)
            if not proposed:
                unresolved.append(issue.issue_id)
                continue
            steps.extend(proposed)
        ordered = self._with_dependencies(self._deduplicate(tuple(steps)))
        costs = [step.estimated_cost for step in ordered]
        total = None if any(value is None for value in costs) else float(sum(value or 0.0 for value in costs))
        payload = {
            "route_id": graph.route_id,
            "cut": replace(cut, runtime_ms=0.0),
            "steps": ordered,
            "unresolved": sorted(set(unresolved)),
        }
        trace = canonical_sha256(payload)
        return RepairPlan(
            plan_id=f"repair:{graph.route_id}:{trace[:12]}",
            cut=cut,
            steps=ordered,
            total_estimated_cost=total,
            fully_executable=bool(ordered) and not unresolved and all(step.executable for step in ordered),
            unresolved_issues=tuple(sorted(set(unresolved))),
            trace_sha256=trace,
        )

    @staticmethod
    def _deduplicate(steps: tuple) -> tuple:
        merged: dict[tuple[str, str, str], object] = {}
        for step in steps:
            key = (step.provider_id, step.operation, step.target)
            previous = merged.get(key)
            if previous is None:
                merged[key] = step
                continue
            merged[key] = replace(
                previous,
                expected_postconditions=tuple(
                    dict.fromkeys((*previous.expected_postconditions, *step.expected_postconditions))
                ),
                verification_checks=tuple(
                    dict.fromkeys((*previous.verification_checks, *step.verification_checks))
                ),
                parameters={
                    **previous.parameters,
                    "issue_ids": tuple(
                        dict.fromkeys(
                            (
                                *previous.parameters.get("issue_ids", (previous.parameters.get("issue_id"),)),
                                step.parameters.get("issue_id"),
                            )
                        )
                    ),
                },
            )
        return tuple(merged.values())

    @staticmethod
    def _with_dependencies(steps: tuple) -> tuple:
        ordered = sorted(steps, key=lambda item: (item.target, item.provider_id, item.step_id))
        result = []
        previous: str | None = None
        for step in ordered:
            dependencies = step.depends_on or ((previous,) if previous else ())
            updated = replace(step, depends_on=tuple(value for value in dependencies if value))
            result.append(updated)
            previous = updated.step_id
        return tuple(result)
