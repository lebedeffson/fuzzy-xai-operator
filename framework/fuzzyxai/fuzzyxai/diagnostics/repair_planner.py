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
                atom.subject_id
                in {
                    subject.split(":", 1)[1] if subject.startswith(("node:", "edge:")) else subject
                    for subject in (*issue.source_nodes, *issue.affected_nodes, *issue.affected_edges)
                }
                or atom.subject_kind == "contract"
                and atom.subject_id == issue.violated_contract
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
            selected_targets = {
                (atom.subject_kind, atom.subject_id)
                for atom in cut.defect_atoms
            }
            selected_source_targets = {
                item for item in selected_targets if item[0] in {"node", "edge"}
            }
            filtered = tuple(
                step
                for step in proposed
                if (step.target.subject_kind, step.target.subject_id)
                in selected_source_targets
            )
            if selected_source_targets:
                proposed = filtered
            if not proposed:
                unresolved.append(issue.issue_id)
                continue
            steps.extend(proposed)
        ordered = self._with_dependencies(
            self._deduplicate(tuple(steps)),
            graph,
        )
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
            key = (
                step.provider_id,
                step.operation,
                step.target.subject_kind,
                step.target.subject_id,
            )
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
                    "contract_ids": tuple(
                        dict.fromkeys(
                            (
                                *previous.parameters.get(
                                    "contract_ids",
                                    (previous.parameters.get("contract_id"),),
                                ),
                                *step.parameters.get(
                                    "contract_ids",
                                    (step.parameters.get("contract_id"),),
                                ),
                            )
                        )
                    ),
                },
            )
        return tuple(merged.values())

    @staticmethod
    def _with_dependencies(steps: tuple, graph: RouteGraph) -> tuple:
        registered = {
            str(target): tuple(str(item) for item in dependencies)
            for target, dependencies in dict(
                graph.metadata.get("repair_dependencies", {})
            ).items()
        }
        by_target = {step.target.subject_id: step for step in steps}
        ordered: list[object] = []
        pending = {
            step.step_id: step for step in steps
        }
        while pending:
            ready = [
                step
                for step in pending.values()
                if all(
                    dependency not in by_target
                    or by_target[dependency].step_id not in pending
                    for dependency in registered.get(
                        step.target.subject_id,
                        (),
                    )
                )
            ]
            if not ready:
                ready = [min(pending.values(), key=lambda item: item.step_id)]
            for step in sorted(
                ready,
                key=lambda item: (item.target, item.provider_id, item.step_id),
            ):
                ordered.append(step)
                pending.pop(step.step_id)
        result = []
        previous: str | None = None
        for step in ordered:
            dependencies = tuple(
                by_target[target].step_id
                for target in registered.get(step.target.subject_id, ())
                if target in by_target
            )
            dependencies = (
                step.depends_on
                or dependencies
                or ((previous,) if previous else ())
            )
            updated = replace(step, depends_on=tuple(value for value in dependencies if value))
            result.append(updated)
            previous = updated.step_id
        return tuple(result)
