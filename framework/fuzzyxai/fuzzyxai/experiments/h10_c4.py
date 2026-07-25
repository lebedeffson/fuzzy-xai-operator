from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
import tracemalloc
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter, perf_counter_ns

from fuzzyxai.diagnostics import (
    Contract,
    DefectAtom,
    DiagnosticCut,
    DiagnosticValidator,
    RepairExecutionContext,
    RepairExecutor,
    RepairPlan,
    RepairStep,
    RouteEdge,
    RouteGraph,
    RouteNode,
    RouteRecertifier,
)
from fuzzyxai.diagnostics.repair_registry import (
    RepairProviderRegistry,
    StepVerification,
)
from fuzzyxai.repair import (
    CostCalibration,
    CostWeights,
    RepairAction,
    StrategyPlan,
    action_cost,
    enumerate_valid_repair_sets,
    run_cost_sensitivity,
    select_first_valid,
    select_global_minimum_cut,
    select_local_greedy,
    select_repair_all,
)

STUDY_ID = "FXAI-H10-C4-OPERATIONAL-UTILITY"
SCENARIO_LABEL = "CONTROLLED_SYNTHETIC_MUTATION"
PRIMARY_WEIGHTS = CostWeights(0.5, 0.5, 0.5)
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 10_420_260_725
FAMILY_SPECS: Mapping[str, tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = {
    "tabular_credit": (
        ("loan_table", "imputer", "credit_model", "shap", "lime", "decision_record"),
        (
            ("loan_table", "imputer"),
            ("imputer", "credit_model"),
            ("credit_model", "shap"),
            ("credit_model", "lime"),
            ("shap", "decision_record"),
            ("lime", "decision_record"),
        ),
    ),
    "tabular_energy": (
        ("meter_table", "gap_filler", "load_model", "shap", "lime", "forecast_record"),
        (
            ("meter_table", "gap_filler"),
            ("gap_filler", "load_model"),
            ("load_model", "shap"),
            ("load_model", "lime"),
            ("shap", "forecast_record"),
            ("lime", "forecast_record"),
        ),
    ),
    "news_text": (
        ("news_document", "tokenizer", "news_model", "shap", "lime", "news_report"),
        (
            ("news_document", "tokenizer"),
            ("tokenizer", "news_model"),
            ("tokenizer", "shap"),
            ("news_model", "shap"),
            ("news_model", "lime"),
            ("shap", "news_report"),
            ("lime", "news_report"),
        ),
    ),
    "reviews_text": (
        ("review_document", "dictionary", "sentiment_model", "shap", "lime", "review_report"),
        (
            ("review_document", "dictionary"),
            ("dictionary", "sentiment_model"),
            ("dictionary", "shap"),
            ("dictionary", "lime"),
            ("sentiment_model", "shap"),
            ("sentiment_model", "lime"),
            ("shap", "review_report"),
            ("lime", "review_report"),
        ),
    ),
    "image_quality": (
        ("image", "geometry_transform", "vision_model", "shap", "lime", "image_report"),
        (
            ("image", "geometry_transform"),
            ("geometry_transform", "vision_model"),
            ("image", "shap"),
            ("vision_model", "shap"),
            ("vision_model", "lime"),
            ("shap", "image_report"),
            ("lime", "image_report"),
        ),
    ),
    "time_series": (
        ("signal", "time_aligner", "sequence_model", "shap", "lime", "signal_report"),
        (
            ("signal", "time_aligner"),
            ("time_aligner", "sequence_model"),
            ("time_aligner", "shap"),
            ("sequence_model", "shap"),
            ("sequence_model", "lime"),
            ("shap", "signal_report"),
            ("lime", "signal_report"),
        ),
    ),
}
MUTATION_FAMILIES = (
    "single_local_unique",
    "independent_multiple",
    "shared_source_downstream",
    "equal_size_different_cost",
    "multiple_equal_optima",
    "downstream_does_not_fix_source",
    "secondary_critical_violation",
    "rollback_required",
    "variable_dependency_fanout",
    "repair_all_touches_more_components",
)


@dataclass(frozen=True)
class H10C4Scenario:
    scenario_id: str
    pipeline_family: str
    mutation_family: str
    seed: int
    source_snapshot_hash: str
    route_graph_hash: str
    registered_violations: tuple[str, ...]
    valid_graph: RouteGraph
    mutated_graph: RouteGraph
    actions: tuple[RepairAction, ...]

    @property
    def obligations(self) -> frozenset[str]:
        return frozenset(self.registered_violations)


@dataclass(frozen=True)
class PlanLevelResult:
    scenario_id: str
    pipeline_family: str
    mutation_family: str
    strategy: str
    selected_cut: tuple[str, ...]
    selected_actions: tuple[str, ...]
    repair_action_count: int
    unique_touched_components: int
    modified_artifact_count: int
    dependency_fanout_sum: int
    dependency_fanout_max: int
    recertification_check_count: int
    execution_time_ms: float
    peak_memory_mb: float
    rollback_count: int
    new_warning_count: int
    new_critical_violation_count: int
    final_route_status: str
    repair_success: bool
    recertification_success: bool
    verifier_consistency: str
    predicted_executable_cost: float
    executed_cost: float
    normalized_executable_cost: float
    postconditions_pass: bool
    before_trace_sha256: str
    after_trace_sha256: str


def _stable_seed(namespace: str, family: str, index: int) -> int:
    payload = f"{namespace}:{family}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _build_valid_graph(family: str, scenario_id: str) -> RouteGraph:
    roles, edge_pairs = FAMILY_SPECS[family]
    nodes = tuple(
        RouteNode(
            node_id=f"{family}:{role}",
            node_type=role,
            component_id=f"{family}.{role}",
            component_version="h10-c4-v1",
            registered_attributes={
                "version": f"{role}-1.0",
                "sample_id": f"{scenario_id}:sample",
                "integrity": "verified",
            },
            observed_attributes={
                "version": f"{role}-1.0",
                "sample_id": f"{scenario_id}:sample",
                "integrity": "verified",
            },
            mandatory=True,
            repairable=True,
            evidence_refs=(f"controlled:{scenario_id}:{role}",),
        )
        for role in roles
    )
    edges = tuple(
        RouteEdge(
            edge_id=f"{family}:edge:{source}->{target}",
            source=f"{family}:{source}",
            target=f"{family}:{target}",
            relation="derived_from",
            mandatory=True,
            registered_contract={"compatible": True},
            observed_contract={"compatible": True},
            repairable=True,
            evidence_refs=(f"controlled:{scenario_id}:edge:{source}->{target}",),
            relation_status="known_valid",
        )
        for source, target in edge_pairs
    )
    contract_roles = (roles[1], roles[2], roles[3], roles[4])
    contracts = tuple(
        Contract(
            contract_id=f"{scenario_id}:obligation:{index}",
            kind="equals",
            subject_id=f"{family}:{role}",
            field="version" if index < 2 else "sample_id",
            expected=(
                f"{role}-1.0"
                if index < 2
                else f"{scenario_id}:sample"
            ),
            severity="error",
            category=("preprocessing", "model", "explainer", "explainer")[index],
            mandatory=True,
            repairable=True,
            evidence_refs=(f"controlled:{scenario_id}:contract:{index}",),
            source_nodes=(f"{family}:{roles[0]}",),
        )
        for index, role in enumerate(contract_roles)
    )
    guard = Contract(
        contract_id=f"{scenario_id}:guard",
        kind="equals",
        subject_id=f"{family}:{roles[-1]}",
        field="integrity",
        expected="verified",
        severity="error",
        category="runtime",
        mandatory=True,
        repairable=True,
        evidence_refs=(f"controlled:{scenario_id}:guard",),
        source_nodes=(f"{family}:{roles[-1]}",),
    )
    return RouteGraph(
        route_id=f"route:{scenario_id}",
        nodes=nodes,
        edges=edges,
        contracts=(*contracts, guard),
        metadata={
            "study_id": STUDY_ID,
            "scenario_label": SCENARIO_LABEL,
            "pipeline_family": family,
        },
    )


def _mutation_indexes(mutation_index: int) -> tuple[int, ...]:
    return {
        0: (0,),
        1: (0, 1),
        2: (2, 3),
        3: (0, 1, 2),
        4: (0, 1, 2, 3),
        5: (0, 2, 3),
        6: (0, 1, 2, 3),
        7: (0, 1, 2, 3),
        8: (0, 1, 2, 3),
        9: (0, 1, 2, 3),
    }[mutation_index]


def _mutate_graph(
    graph: RouteGraph,
    selected_contracts: tuple[Contract, ...],
) -> RouteGraph:
    selected = {contract.contract_id for contract in selected_contracts}
    by_subject = {node.node_id: node for node in graph.nodes}
    for contract in graph.contracts:
        if contract.contract_id not in selected:
            continue
        node = by_subject[contract.subject_id]
        observed = dict(node.observed_attributes)
        observed[str(contract.field)] = f"mutated:{contract.contract_id}"
        by_subject[node.node_id] = replace(node, observed_attributes=observed)
    return replace(graph, nodes=tuple(by_subject[node.node_id] for node in graph.nodes))


def _scenario_actions(
    graph: RouteGraph,
    obligations: tuple[str, ...],
    mutation_index: int,
) -> tuple[RepairAction, ...]:
    roles, _ = FAMILY_SPECS[str(graph.metadata["pipeline_family"])]
    contract_by_id = {contract.contract_id: contract for contract in graph.contracts}
    direct = tuple(
        RepairAction(
            action_id=f"direct:{index}:{contract_id}",
            target_component=contract_by_id[contract_id].subject_id,
            covers=frozenset({contract_id}),
            action_kind="direct_restore",
            dependency_fanout=1 + (index % 2),
            runtime_units=2 + (index % 2),
            rollback_risk=0.04,
            modified_artifacts=1,
            direct_repair=True,
        )
        for index, contract_id in enumerate(obligations)
    )
    source = RepairAction(
        action_id="source:restore",
        target_component=f"{graph.metadata['pipeline_family']}:{roles[0]}",
        covers=frozenset(obligations),
        action_kind="source_restore",
        dependency_fanout=2 + (mutation_index % 5),
        runtime_units=4,
        rollback_risk=0.03,
        modified_artifacts=1,
        direct_repair=False,
    )
    alternatives: list[RepairAction] = [source]
    if len(obligations) >= 3:
        alternatives.append(
            RepairAction(
                action_id="local:partial_fast",
                target_component=f"{graph.metadata['pipeline_family']}:{roles[3]}",
                covers=frozenset(obligations[:-1]),
                action_kind="quick_patch",
                dependency_fanout=0,
                runtime_units=1,
                rollback_risk=0.02,
                modified_artifacts=1,
                direct_repair=False,
            )
        )
    if mutation_index == 4:
        alternatives.append(
            replace(
                source,
                action_id="source:restore_equivalent",
                target_component=f"{graph.metadata['pipeline_family']}:{roles[1]}",
            )
        )
    if mutation_index in {6, 7}:
        alternatives.append(
            RepairAction(
                action_id="unsafe:shortcut",
                target_component=f"{graph.metadata['pipeline_family']}:{roles[-1]}",
                covers=frozenset(obligations),
                action_kind="quick_patch",
                dependency_fanout=0,
                runtime_units=1,
                rollback_risk=1.0,
                modified_artifacts=1,
                direct_repair=False,
                creates_critical_violation=True,
            )
        )
    if mutation_index == 8 and len(direct) >= 2:
        direct = (
            direct[0],
            replace(
                direct[1],
                dependencies=(direct[0].action_id,),
                dependency_fanout=7,
            ),
            *direct[2:],
        )
    return (*direct, *alternatives)


def build_scenarios(
    *,
    split: str,
    scenarios_per_family: int,
) -> tuple[H10C4Scenario, ...]:
    namespace = (
        "H10-C4-DEV-20260725"
        if split == "development"
        else "H10-C4-HELDOUT-20260725"
    )
    scenarios = []
    for family_index, family in enumerate(FAMILY_SPECS):
        for local_index in range(scenarios_per_family):
            global_index = family_index * scenarios_per_family + local_index
            mutation_index = global_index % len(MUTATION_FAMILIES)
            seed = _stable_seed(namespace, family, local_index)
            scenario_id = f"h10-c4:{split}:{family}:{local_index:03d}:{seed:016x}"
            valid = _build_valid_graph(family, scenario_id)
            indexes = _mutation_indexes(mutation_index)
            selected = tuple(valid.contracts[index] for index in indexes)
            mutated = _mutate_graph(valid, selected)
            actions = _scenario_actions(
                valid,
                tuple(contract.contract_id for contract in selected),
                mutation_index,
            )
            if not DiagnosticValidator().validate(valid).valid:
                raise AssertionError(f"invalid source graph: {scenario_id}")
            if DiagnosticValidator().validate(mutated).valid:
                raise AssertionError(f"mutation did not invalidate graph: {scenario_id}")
            scenarios.append(
                H10C4Scenario(
                    scenario_id=scenario_id,
                    pipeline_family=family,
                    mutation_family=MUTATION_FAMILIES[mutation_index],
                    seed=seed,
                    source_snapshot_hash=valid.trace_sha256,
                    route_graph_hash=mutated.trace_sha256,
                    registered_violations=tuple(
                        contract.contract_id for contract in selected
                    ),
                    valid_graph=valid,
                    mutated_graph=mutated,
                    actions=actions,
                )
            )
    return tuple(scenarios)


def _runtime_workload(action: RepairAction) -> None:
    digest = action.action_id.encode()
    for index in range(action.runtime_units * 80):
        digest = hashlib.sha256(digest + index.to_bytes(2, "little")).digest()


def calibrate_runtime(
    development: tuple[H10C4Scenario, ...],
) -> CostCalibration:
    by_kind: dict[str, list[RepairAction]] = {}
    for scenario in development:
        for action in scenario.actions:
            by_kind.setdefault(action.action_kind, []).append(action)
    medians: dict[str, float] = {}
    for kind, actions in sorted(by_kind.items()):
        representative = actions[0]
        samples = []
        for _ in range(21):
            started = perf_counter_ns()
            _runtime_workload(representative)
            samples.append((perf_counter_ns() - started) / 1_000_000)
        medians[kind] = statistics.median(samples)
    return CostCalibration(
        runtime_ms_by_kind=medians,
        runtime_scale_ms=max(statistics.median(medians.values()), 1e-9),
    )


def _development_action_registry_sha256(
    development: tuple[H10C4Scenario, ...],
) -> str:
    return _canonical_hash(
        [
            asdict(action)
            for scenario in development
            for action in scenario.actions
        ]
    )


def _load_or_create_calibration(
    development: tuple[H10C4Scenario, ...],
    path: Path,
) -> CostCalibration:
    registry_sha256 = _development_action_registry_sha256(development)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["development_action_registry_sha256"] != registry_sha256:
            raise AssertionError("development calibration action registry changed")
        return CostCalibration(
            runtime_ms_by_kind={
                key: float(value)
                for key, value in payload["runtime_ms_by_kind"].items()
            },
            runtime_scale_ms=float(payload["runtime_scale_ms"]),
            fitted_split=str(payload["fitted_split"]),
        )
    calibration = calibrate_runtime(development)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                **asdict(calibration),
                "development_action_registry_sha256": registry_sha256,
                "held_out_used_for_calibration": False,
                "status": "FROZEN_BEFORE_HELD_OUT_EXECUTION",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return calibration


class _OperationalRepairProvider:
    provider_id = "h10_c4.execute"

    def verify(
        self,
        before: RouteGraph,
        after: RouteGraph,
        step: RepairStep,
    ) -> StepVerification:
        before_result = DiagnosticValidator().validate(before)
        after_result = DiagnosticValidator().validate(after)
        before_issues = {issue.issue_id for issue in before_result.issues}
        after_issues = {issue.issue_id for issue in after_result.issues}
        expected = {
            f"issue:{contract_id}"
            for contract_id in step.parameters.get("contract_ids", ())
        }
        resolved = not expected.intersection(after_issues)
        new_critical = after_issues - before_issues
        changed = before.trace_sha256 != after.trace_sha256
        return StepVerification(
            passed=changed and resolved and not new_critical,
            checks=(
                {"check": "route_changed", "passed": changed},
                {
                    "check": "expected_contracts_satisfied",
                    "passed": resolved,
                    "contracts": tuple(step.parameters.get("contract_ids", ())),
                },
                {
                    "check": "no_new_critical_violations",
                    "passed": not new_critical,
                    "new_issues": tuple(sorted(new_critical)),
                },
            ),
        )


def _apply_action(
    graph: RouteGraph,
    action: RepairAction,
    step: RepairStep,
) -> RouteGraph:
    _runtime_workload(action)
    contract_ids = set(step.parameters["contract_ids"])
    nodes = {node.node_id: node for node in graph.nodes}
    for contract in graph.contracts:
        if contract.contract_id not in contract_ids:
            continue
        node = nodes[contract.subject_id]
        observed = dict(node.observed_attributes)
        observed[str(contract.field)] = contract.expected
        nodes[node.node_id] = replace(node, observed_attributes=observed)
    if action.creates_critical_violation:
        guard = next(contract for contract in graph.contracts if contract.contract_id.endswith(":guard"))
        node = nodes[guard.subject_id]
        observed = dict(node.observed_attributes)
        observed[str(guard.field)] = "corrupted_by_repair"
        nodes[node.node_id] = replace(node, observed_attributes=observed)
    return replace(graph, nodes=tuple(nodes[node.node_id] for node in graph.nodes))


def _repair_plan(
    scenario: H10C4Scenario,
    strategy: StrategyPlan,
    cost: Callable[[RepairAction], float],
) -> RepairPlan:
    action_by_id = {action.action_id: action for action in scenario.actions}
    steps = []
    for action_id in strategy.action_ids:
        action = action_by_id[action_id]
        steps.append(
            RepairStep(
                step_id=f"step:{action.action_id}",
                title=f"Execute {action.action_id}",
                target=DefectAtom(
                    "node",
                    action.target_component,
                    None,
                    "controlled_structural_mutation",
                    True,
                    cost(action),
                ),
                provider_id="h10_c4.execute",
                operation=f"h10_c4:{action.action_id}",
                parameters={
                    "contract_ids": tuple(sorted(action.covers)),
                    "scenario_id": scenario.scenario_id,
                },
                preconditions=("controlled_snapshot_available",),
                depends_on=tuple(f"step:{item}" for item in action.dependencies),
                expected_postconditions=tuple(
                    f"contract_satisfied:{item}" for item in sorted(action.covers)
                ),
                verification_checks=(
                    "expected_contracts_satisfied",
                    "no_new_critical_violations",
                    "full_recertification",
                ),
                fallback_step_ids=(),
                rollback_operation=None,
                estimated_cost=cost(action),
                requires_human_approval=False,
                executable=True,
            )
        )
    cut = DiagnosticCut(
        defect_atoms=tuple(step.target for step in steps),
        affected_nodes=tuple(sorted({step.target.subject_id for step in steps})),
        total_cost=strategy.predicted_cost,
        optimal=strategy.strategy == "O_GLOBAL",
        solver=("exhaustive_global" if strategy.strategy == "O_GLOBAL" else strategy.strategy),
        covered_obligations=strategy.covered_obligations,
        uncovered_obligations=tuple(
            sorted(scenario.obligations - set(strategy.covered_obligations))
        ),
        equivalent_optimal_cuts=(),
        runtime_ms=0.0,
        enumeration_complete=strategy.strategy == "O_GLOBAL",
        truncated=False,
        lower_bound_count=len(strategy.equivalent_optimal_plans),
        optimality_proven=strategy.strategy == "O_GLOBAL",
    )
    return RepairPlan(
        plan_id=f"{scenario.scenario_id}:{strategy.strategy}",
        cut=cut,
        steps=tuple(steps),
        total_estimated_cost=strategy.predicted_cost,
        fully_executable=strategy.feasible,
        unresolved_issues=cut.uncovered_obligations,
        trace_sha256=_canonical_hash(
            {
                "scenario_id": scenario.scenario_id,
                "strategy": strategy.strategy,
                "actions": strategy.action_ids,
            }
        ),
    )


def execute_strategy(
    scenario: H10C4Scenario,
    strategy: StrategyPlan,
    calibration: CostCalibration,
    *,
    repair_all_cost: float,
) -> PlanLevelResult:
    action_by_id = {action.action_id: action for action in scenario.actions}
    primary_cost = lambda action: action_cost(
        action,
        "hybrid",
        calibration,
        PRIMARY_WEIGHTS,
    )
    plan = _repair_plan(scenario, strategy, primary_cost)
    registry = RepairProviderRegistry(providers=[_OperationalRepairProvider()])
    executor = RepairExecutor(registry)
    handlers = {
        f"h10_c4:{action.action_id}": (
            lambda graph, step, selected=action: _apply_action(graph, selected, step)
        )
        for action in scenario.actions
    }
    context = RepairExecutionContext(
        handlers=handlers,
        allow_external_changes=True,
        satisfied_preconditions=frozenset({"controlled_snapshot_available"}),
    )
    tracemalloc.start()
    started = perf_counter()
    after, step_results = executor.execute(scenario.mutated_graph, plan, context)
    elapsed_ms = (perf_counter() - started) * 1000
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    recertification = RouteRecertifier().recertify(
        scenario.mutated_graph,
        after,
        plan,
        step_results,
    )
    final_validation = DiagnosticValidator().validate(after)
    attempted = tuple(
        action_by_id[result.step_id.removeprefix("step:")]
        for result in step_results
        if result.status not in {"dependency_failed", "not_approved", "handler_unavailable"}
    )
    rollback_actions = tuple(
        action_by_id[result.step_id.removeprefix("step:")]
        for result in step_results
        if result.rollback_verified
    )
    executed_cost = sum(primary_cost(action) for action in attempted) + sum(
        primary_cost(action) for action in rollback_actions
    )
    completed_ids = {
        result.step_id.removeprefix("step:")
        for result in step_results
        if result.status == "completed"
    }
    completed_actions = tuple(
        action for action in scenario.actions if action.action_id in completed_ids
    )
    touched = {
        component
        for action in completed_actions
        for component in (
            action.target_component,
            *action.additional_touched_components,
        )
    }
    postconditions_pass = all(
        result.status == "completed"
        and all(bool(check.get("passed")) for check in result.verification)
        for result in step_results
    )
    verifier_consistency = (
        "PASS"
        if final_validation.valid == recertification.route_valid_after
        else "FAIL"
    )
    repair_success = (
        postconditions_pass
        and not recertification.new_critical_issues
        and recertification.status == "full_success"
        and verifier_consistency == "PASS"
    )
    check_count = sum(len(result.verification) for result in step_results) + len(
        final_validation.checked_contracts
    )
    return PlanLevelResult(
        scenario_id=scenario.scenario_id,
        pipeline_family=scenario.pipeline_family,
        mutation_family=scenario.mutation_family,
        strategy=strategy.strategy,
        selected_cut=strategy.action_ids,
        selected_actions=strategy.action_ids,
        repair_action_count=len(attempted),
        unique_touched_components=len(touched),
        modified_artifact_count=sum(action.modified_artifacts for action in attempted),
        dependency_fanout_sum=sum(action.dependency_fanout for action in attempted),
        dependency_fanout_max=max(
            (action.dependency_fanout for action in attempted),
            default=0,
        ),
        recertification_check_count=check_count,
        execution_time_ms=elapsed_ms,
        peak_memory_mb=peak_bytes / (1024 * 1024),
        rollback_count=len(rollback_actions),
        new_warning_count=0,
        new_critical_violation_count=len(recertification.new_critical_issues),
        final_route_status=recertification.status,
        repair_success=repair_success,
        recertification_success=recertification.status == "full_success",
        verifier_consistency=verifier_consistency,
        predicted_executable_cost=strategy.predicted_cost,
        executed_cost=executed_cost,
        normalized_executable_cost=executed_cost / max(repair_all_cost, 1e-12),
        postconditions_pass=postconditions_pass,
        before_trace_sha256=scenario.mutated_graph.trace_sha256,
        after_trace_sha256=after.trace_sha256,
    )


def _strategy_plans(
    scenario: H10C4Scenario,
    calibration: CostCalibration,
) -> tuple[StrategyPlan, ...]:
    cost = lambda action: action_cost(action, "hybrid", calibration, PRIMARY_WEIGHTS)
    return (
        select_repair_all(scenario.actions, scenario.obligations, cost),
        select_first_valid(scenario.actions, scenario.obligations, cost),
        select_local_greedy(scenario.actions, scenario.obligations, cost),
        select_global_minimum_cut(scenario.actions, scenario.obligations, cost),
    )


def run_scenarios(
    scenarios: tuple[H10C4Scenario, ...],
    calibration: CostCalibration,
) -> tuple[PlanLevelResult, ...]:
    rows = []
    for scenario in scenarios:
        plans = _strategy_plans(scenario, calibration)
        repair_all = plans[0].predicted_cost
        for plan in plans:
            rows.append(
                execute_strategy(
                    scenario,
                    plan,
                    calibration,
                    repair_all_cost=repair_all,
                )
            )
    return tuple(rows)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _paired_bootstrap(
    differences: list[float],
    *,
    index_stream: tuple[tuple[int, ...], ...],
    index_stream_sha256: str,
) -> dict[str, object]:
    n = len(differences)
    estimates = [
        sum(differences[index] for index in indexes) / n
        for indexes in index_stream
    ]
    estimates.sort()
    lower = estimates[int(0.025 * BOOTSTRAP_ITERATIONS)]
    upper = estimates[min(BOOTSTRAP_ITERATIONS - 1, int(0.975 * BOOTSTRAP_ITERATIONS))]
    less_equal = sum(value <= 0.0 for value in estimates)
    greater_equal = sum(value >= 0.0 for value in estimates)
    p_value = min(
        1.0,
        2.0
        * min(less_equal + 1, greater_equal + 1)
        / (BOOTSTRAP_ITERATIONS + 1),
    )
    return {
        "n": n,
        "mean_difference": statistics.mean(differences),
        "median_difference": statistics.median(differences),
        "ci_95": [lower, upper],
        "bootstrap_p_two_sided": p_value,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_index_stream_sha256": index_stream_sha256,
    }


def _holm(p_values: list[float]) -> list[float]:
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(p_values) - rank) * value))
        adjusted[index] = running
    return adjusted


def analyze(
    results: tuple[PlanLevelResult, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    by_key = {(row.scenario_id, row.strategy): row for row in results}
    scenario_ids = sorted({row.scenario_id for row in results})
    rng = random.Random(BOOTSTRAP_SEED)
    index_stream = tuple(
        tuple(rng.randrange(len(scenario_ids)) for _ in scenario_ids)
        for _ in range(BOOTSTRAP_ITERATIONS)
    )
    stream_hasher = hashlib.sha256()
    for indexes in index_stream:
        stream_hasher.update(
            b"".join(index.to_bytes(2, "little") for index in indexes)
        )
    index_stream_sha256 = stream_hasher.hexdigest()
    comparisons = []
    for baseline in ("B_ALL", "B_FIRST", "B_GREEDY"):
        differences = [
            by_key[(scenario_id, "O_GLOBAL")].normalized_executable_cost
            - by_key[(scenario_id, baseline)].normalized_executable_cost
            for scenario_id in scenario_ids
        ]
        summary = _paired_bootstrap(
            differences,
            index_stream=index_stream,
            index_stream_sha256=index_stream_sha256,
        )
        comparisons.append(
            {
                "comparison": f"O_GLOBAL_vs_{baseline}",
                "endpoint": "normalized_executable_cost",
                **summary,
            }
        )
    adjusted = _holm(
        [float(row["bootstrap_p_two_sided"]) for row in comparisons]
    )
    for row, value in zip(comparisons, adjusted):
        row["holm_p"] = value

    descriptive = []
    for strategy in ("B_ALL", "B_FIRST", "B_GREEDY", "O_GLOBAL"):
        strategy_rows = [row for row in results if row.strategy == strategy]
        for metric in (
            "normalized_executable_cost",
            "repair_action_count",
            "unique_touched_components",
            "recertification_check_count",
            "execution_time_ms",
            "new_critical_violation_count",
            "rollback_count",
        ):
            values = [float(getattr(row, metric)) for row in strategy_rows]
            descriptive.append(
                {
                    "strategy": strategy,
                    "metric": metric,
                    "n": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "p90": _quantile(values, 0.9),
                    "worst_case": max(values),
                }
            )

    global_rows = [row for row in results if row.strategy == "O_GLOBAL"]
    baseline_rows = {
        strategy: [row for row in results if row.strategy == strategy]
        for strategy in ("B_ALL", "B_FIRST", "B_GREEDY")
    }
    primary_pass = all(
        float(row["mean_difference"]) < 0.0
        and float(row["ci_95"][1]) < 0.0
        and float(row["holm_p"]) < 0.05
        for row in comparisons
    )
    success_global = sum(row.repair_success for row in global_rows) / len(global_rows)
    success_baselines = {
        strategy: sum(row.repair_success for row in rows) / len(rows)
        for strategy, rows in baseline_rows.items()
    }
    secondary_metrics = (
        "repair_action_count",
        "unique_touched_components",
        "recertification_check_count",
        "execution_time_ms",
    )
    strongest = max(
        success_baselines,
        key=lambda item: (
            success_baselines[item],
            -statistics.mean(
                row.normalized_executable_cost for row in baseline_rows[item]
            ),
        ),
    )
    secondary_improvements = {
        metric: (
            statistics.mean(float(getattr(row, metric)) for row in global_rows)
            < statistics.mean(
                float(getattr(row, metric)) for row in baseline_rows[strongest]
            )
        )
        for metric in secondary_metrics
    }
    supported = (
        primary_pass
        and all(success_global >= value for value in success_baselines.values())
        and sum(row.new_critical_violation_count for row in global_rows) == 0
        and any(secondary_improvements.values())
    )
    status = {
        "protocol_id": "h10-c4-operational-utility-v1",
        "scenario_count": len(scenario_ids),
        "strategy_rows": len(results),
        "repair_success_global_cut": success_global,
        "repair_success_baselines": success_baselines,
        "new_critical_violations_global_cut": sum(
            row.new_critical_violation_count for row in global_rows
        ),
        "strongest_successful_baseline": strongest,
        "secondary_improvements": secondary_improvements,
        "primary_comparisons_pass": primary_pass,
        "status": (
            "H10_C4_SUPPORTED"
            if supported
            else "OPERATIONAL_ADVANTAGE_NOT_CONFIRMED"
        ),
        "claim_scope": "controlled_structural_mutations_only",
    }
    return comparisons, descriptive, status


def _family_results(
    results: tuple[PlanLevelResult, ...],
) -> list[dict[str, object]]:
    rows = []
    by_key = {(row.scenario_id, row.strategy): row for row in results}
    for family in sorted({row.pipeline_family for row in results}):
        scenario_ids = sorted(
            {
                row.scenario_id
                for row in results
                if row.pipeline_family == family
            }
        )
        for strategy in ("B_ALL", "B_FIRST", "B_GREEDY", "O_GLOBAL"):
            selected = [by_key[(scenario_id, strategy)] for scenario_id in scenario_ids]
            rows.append(
                {
                    "pipeline_family": family,
                    "strategy": strategy,
                    "scenario_count": len(selected),
                    "repair_success_rate": sum(
                        row.repair_success for row in selected
                    )
                    / len(selected),
                    "mean_normalized_executable_cost": statistics.mean(
                        row.normalized_executable_cost for row in selected
                    ),
                    "mean_repair_action_count": statistics.mean(
                        row.repair_action_count for row in selected
                    ),
                    "mean_touched_component_count": statistics.mean(
                        row.unique_touched_components for row in selected
                    ),
                    "mean_recertification_check_count": statistics.mean(
                        row.recertification_check_count for row in selected
                    ),
                    "mean_execution_time_ms": statistics.mean(
                        row.execution_time_ms for row in selected
                    ),
                }
            )
        for baseline in ("B_ALL", "B_FIRST", "B_GREEDY"):
            differences = [
                by_key[(scenario_id, "O_GLOBAL")].normalized_executable_cost
                - by_key[(scenario_id, baseline)].normalized_executable_cost
                for scenario_id in scenario_ids
            ]
            rows.append(
                {
                    "pipeline_family": family,
                    "strategy": f"O_GLOBAL_vs_{baseline}",
                    "scenario_count": len(differences),
                    "repair_success_rate": "",
                    "mean_normalized_executable_cost": statistics.mean(differences),
                    "mean_repair_action_count": "",
                    "mean_touched_component_count": "",
                    "mean_recertification_check_count": "",
                    "mean_execution_time_ms": "",
                }
            )
    return rows


def _csv_value(value: object) -> object:
    if isinstance(value, (tuple, list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields = list(materialized[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(value) for key, value in row.items()}
            for row in materialized
        )


def _h10_c3_overlap(scenarios: tuple[H10C4Scenario, ...], root: Path) -> dict[str, object]:
    old_ids: set[str] = set()
    old_state_hashes: set[str] = set()
    old_seed_tokens: set[str] = set()
    for path in (
        root / "artifacts/h10_c3_r4/results/development.csv",
        root / "artifacts/h10_c3_r4/results/protocol_validation.csv",
        root / "artifacts/h10_c3_r4/results/sealed.csv",
    ):
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                old_ids.add(row["case_id"])
                old_state_hashes.update(
                    {
                        row["before_trace_sha256"],
                        row["after_trace_sha256"],
                    }
                )
                old_seed_tokens.update(
                    token for token in row["case_id"].split(":") if token.isdigit()
                )
    new_ids = {scenario.scenario_id for scenario in scenarios}
    new_state_hashes = {
        value
        for scenario in scenarios
        for value in (
            scenario.source_snapshot_hash,
            scenario.route_graph_hash,
        )
    }
    new_seed_tokens = {str(scenario.seed) for scenario in scenarios}
    return {
        "h10_c3_case_id_count": len(old_ids),
        "h10_c3_serialized_state_hash_count": len(old_state_hashes),
        "h10_c4_case_id_count": len(new_ids),
        "h10_c4_serialized_state_hash_count": len(new_state_hashes),
        "case_id_intersection": sorted(old_ids & new_ids),
        "serialized_state_hash_intersection": sorted(
            old_state_hashes & new_state_hashes
        ),
        "seed_token_intersection": sorted(old_seed_tokens & new_seed_tokens),
        "seed_namespaces": {
            "h10_c3": "H10-C3-R4 registered split namespaces",
            "h10_c4": "H10-C4-HELDOUT-20260725",
        },
        "status": (
            "PASS"
            if (
                not (old_ids & new_ids)
                and not (old_state_hashes & new_state_hashes)
                and not (old_seed_tokens & new_seed_tokens)
            )
            else "FAIL"
        ),
    }


def run_experiment(root: Path) -> dict[str, object]:
    development = build_scenarios(split="development", scenarios_per_family=4)
    held_out = build_scenarios(split="held_out", scenarios_per_family=20)
    if len(development) != 24 or len(held_out) != 120:
        raise AssertionError("registered H10-C4 design size mismatch")
    overlap = _h10_c3_overlap(held_out, root)
    if overlap["status"] != "PASS":
        raise AssertionError(f"H10-C3 overlap audit failed: {overlap}")

    result_root = root / "results/h10_c4"
    report_root = root / "reports/h10_c4"
    result_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    calibration = _load_or_create_calibration(
        development,
        result_root / "DEVELOPMENT_COST_CALIBRATION.json",
    )
    results = run_scenarios(held_out, calibration)
    comparisons, descriptive, status = analyze(results)
    family_results = _family_results(results)

    manifest_rows = []
    sensitivity_rows = []
    stability_rows = []
    cost_model_rows = []
    cost = lambda action: action_cost(action, "hybrid", calibration, PRIMARY_WEIGHTS)
    for scenario in held_out:
        valid_sets = enumerate_valid_repair_sets(scenario.actions, scenario.obligations)
        optimal = select_global_minimum_cut(scenario.actions, scenario.obligations, cost)
        manifest_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "pipeline_family": scenario.pipeline_family,
                "mutation_family": scenario.mutation_family,
                "scenario_type": SCENARIO_LABEL,
                "seed": scenario.seed,
                "source_snapshot_hash": scenario.source_snapshot_hash,
                "route_graph_hash": scenario.route_graph_hash,
                "registered_violations": scenario.registered_violations,
                "candidate_repairs": tuple(
                    action.action_id for action in scenario.actions
                ),
                "valid_repair_sets": tuple(
                    tuple(action.action_id for action in plan)
                    for plan in valid_sets
                ),
                "optimal_repair_sets": optimal.equivalent_optimal_plans,
            }
        )
        sensitivity = run_cost_sensitivity(
            scenario.actions,
            scenario.obligations,
            calibration,
        )
        repair_all_primary = select_repair_all(
            scenario.actions,
            scenario.obligations,
            cost,
        ).predicted_cost
        execution_cache: dict[tuple[str, ...], PlanLevelResult] = {}
        for row in sensitivity:
            execution = execution_cache.get(row.selected_cut)
            if execution is None:
                selected = StrategyPlan(
                    strategy="O_GLOBAL_SENSITIVITY",
                    action_ids=row.selected_cut,
                    predicted_cost=row.selected_cut_cost,
                    covered_obligations=tuple(sorted(scenario.obligations)),
                    feasible=row.repair_success,
                    equivalent_optimal_plans=row.alternative_optimal_cuts,
                )
                execution = execute_strategy(
                    scenario,
                    selected,
                    calibration,
                    repair_all_cost=repair_all_primary,
                )
                execution_cache[row.selected_cut] = execution
            sensitivity_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "pipeline_family": scenario.pipeline_family,
                    **asdict(row),
                    "repair_success": execution.repair_success,
                    "recertification_status": execution.final_route_status,
                    "new_critical_violation_count": (
                        execution.new_critical_violation_count
                    ),
                }
            )
        stable = sum(not row.selection_changed for row in sensitivity) / len(sensitivity)
        stability_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "pipeline_family": scenario.pipeline_family,
                "selection_stability_rate": stable,
                "top_k_stability_rate": stable,
                "mean_cost_regret": statistics.mean(
                    row.cost_regret for row in sensitivity
                ),
                "worst_case_cost_regret": max(
                    row.cost_regret for row in sensitivity
                ),
                "number_of_weight_configs": len(sensitivity),
                "stability_threshold": 0.8,
                "stable": stable >= 0.8,
            }
        )
        for model in ("uniform", "runtime", "dependency_weighted", "hybrid"):
            model_cost = lambda action, selected=model: action_cost(
                action,
                selected,
                calibration,
                PRIMARY_WEIGHTS,
            )
            selected = select_global_minimum_cut(
                scenario.actions,
                scenario.obligations,
                model_cost,
            )
            execution = execution_cache.get(selected.action_ids)
            if execution is None:
                execution = execute_strategy(
                    scenario,
                    selected,
                    calibration,
                    repair_all_cost=repair_all_primary,
                )
                execution_cache[selected.action_ids] = execution
            cost_model_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "pipeline_family": scenario.pipeline_family,
                    "cost_model": model,
                    "selected_cut": selected.action_ids,
                    "selected_cut_cost": selected.predicted_cost,
                    "alternative_optimal_cuts": selected.equivalent_optimal_plans,
                    "repair_success": execution.repair_success,
                    "recertification_status": execution.final_route_status,
                    "new_critical_violation_count": (
                        execution.new_critical_violation_count
                    ),
                    "execution_time_ms": execution.execution_time_ms,
                }
            )

    _write_csv(result_root / "SCENARIO_MANIFEST.csv", manifest_rows)
    _write_csv(
        result_root / "PLAN_LEVEL_RESULTS.csv",
        (asdict(row) for row in results),
    )
    _write_csv(result_root / "STRATEGY_COMPARISON.csv", descriptive)
    _write_csv(result_root / "PIPELINE_FAMILY_RESULTS.csv", family_results)
    _write_csv(result_root / "COST_MODEL_COMPARISON.csv", cost_model_rows)
    _write_csv(result_root / "COST_SENSITIVITY.csv", sensitivity_rows)
    _write_csv(result_root / "BOOTSTRAP_INTERVALS.csv", comparisons)
    _write_csv(
        result_root / "HOLM_CORRECTION.csv",
        (
            {
                "comparison": row["comparison"],
                "raw_p": row["bootstrap_p_two_sided"],
                "holm_p": row["holm_p"],
            }
            for row in comparisons
        ),
    )
    _write_csv(result_root / "SELECTION_STABILITY.csv", stability_rows)
    (result_root / "H10_C4_FINAL_STATUS.json").write_text(
        json.dumps(
            {
                **status,
                "selection_stability_rate": sum(
                    row["stable"] for row in stability_rows
                )
                / len(stability_rows),
                "cost_weight_configurations": 48,
                "overlap_audit": overlap,
                "runtime_calibration": asdict(calibration),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_reports(
        report_root,
        status=status,
        comparisons=comparisons,
        descriptive=descriptive,
        stability_rows=stability_rows,
        overlap=overlap,
        calibration=calibration,
    )
    return status


def _write_reports(
    root: Path,
    *,
    status: Mapping[str, object],
    comparisons: list[dict[str, object]],
    descriptive: list[dict[str, object]],
    stability_rows: list[dict[str, object]],
    overlap: Mapping[str, object],
    calibration: CostCalibration,
) -> None:
    comparison_lines = "\n".join(
        (
            f"- `{row['comparison']}`: mean O_GLOBAL-baseline "
            f"`{row['mean_difference']:.6f}`, 95% CI "
            f"`[{row['ci_95'][0]:.6f}; {row['ci_95'][1]:.6f}]`, "
            f"Holm p `{row['holm_p']:.6g}`."
        )
        for row in comparisons
    )
    (root / "OPERATIONAL_UTILITY_REPORT.md").write_text(
        "# H10-C4 Operational Utility Report\n\n"
        f"Status: `{status['status']}`.\n\n"
        "Scope: controlled structural mutations only. This result does not "
        "measure engineer time, human utility, natural incidents, or "
        "organizational cost.\n\n"
        f"{comparison_lines}\n\n"
        f"Global repair success: `{status['repair_success_global_cut']}`. "
        f"New critical violations: "
        f"`{status['new_critical_violations_global_cut']}`.\n",
        encoding="utf-8",
    )
    stable_rate = sum(bool(row["stable"]) for row in stability_rows) / len(
        stability_rows
    )
    (root / "COST_CALIBRATION_AND_LIMITS.md").write_text(
        "# Cost Calibration and Limits\n\n"
        "Runtime normalization was fitted on 24 development scenarios only. "
        "Held-out outcomes were not used for calibration.\n\n"
        f"- runtime scale: `{calibration.runtime_scale_ms:.9f} ms`;\n"
        f"- calibrated action kinds: `{json.dumps(calibration.runtime_ms_by_kind, sort_keys=True)}`;\n"
        f"- scenarios meeting the 80% stability threshold: `{stable_rate:.6f}`;\n"
        "- runtime is machine execution time, not engineer time;\n"
        "- dependency fan-out is structural impact, not organizational expense.\n",
        encoding="utf-8",
    )
    (root / "OPERATOR_COMPOSITION_TRACE.md").write_text(
        "# Operator Composition Trace\n\n"
        "The executable route is:\n\n"
        "```text\n"
        "CollectExplanationArtifact\n"
        "-> ValidateArtifactProvenance\n"
        "-> ValidateExplainerContract\n"
        "-> BuildDiagnosticGraph\n"
        "-> SelectRepairCut\n"
        "-> ExecuteRepairPlan\n"
        "-> RecertifyRoute\n"
        "```\n\n"
        "Composition is sequential and contract-checked. Incompatible "
        "input/output contracts fail with `OPERATOR_CONTRACT_MISMATCH`. SHAP "
        "and LIME artifacts are routed through a shared provenance check but "
        "their attribution values are not mathematically combined.\n\n"
        "A complete symbolic operator algebra with general closure, identity, "
        "and inverse operators is not implemented.\n",
        encoding="utf-8",
    )
    negatives = []
    if status["status"] != "H10_C4_SUPPORTED":
        negatives.append("The preregistered operational-advantage rule did not pass.")
    if stable_rate < 1.0:
        negatives.append(
            "Some scenarios are sensitive to the registered hybrid-cost weights."
        )
    (root / "NEGATIVE_RESULTS.md").write_text(
        "# H10-C4 Negative and Limiting Results\n\n"
        + (
            "\n".join(f"- {item}" for item in negatives)
            if negatives
            else "- No preregistered primary gate failed. Scope limitations remain."
        )
        + "\n\n"
        "- No natural software incident was evaluated.\n"
        "- No human or expert evaluation was conducted.\n"
        "- Machine runtime is not engineer labor time.\n",
        encoding="utf-8",
    )
    (root / "REPRODUCTION_REPORT.md").write_text(
        "# H10-C4 Reproduction\n\n"
        "```bash\n"
        "make h10-c4-test\n"
        "make h10-c4-run\n"
        "make h10-c4-verify\n"
        "```\n\n"
        f"H10-C3 overlap audit: `{overlap['status']}`. "
        "The protocol lock predates result generation.\n",
        encoding="utf-8",
    )
    (root / "H10_C3_IMMUTABILITY_REPORT.md").write_text(
        "# H10-C3 Immutability Report\n\n"
        "The H10-C4 branch starts from postopen commit "
        "`5da8d1beec7681f6b18794cf4001decf4ceb3ea2`. Before H10-C4 execution, "
        "54 H10-C3 R4 files were recorded in "
        "`protocol/h10_c4/H10_C3_BASELINE_SHA256SUMS`.\n\n"
        "Final verification is performed by `make h10-c4-verify`; any changed "
        "H10-C3 evidence file fails the gate.\n",
        encoding="utf-8",
    )


def verify_outputs(root: Path) -> dict[str, object]:
    required = (
        "SCENARIO_MANIFEST.csv",
        "PLAN_LEVEL_RESULTS.csv",
        "STRATEGY_COMPARISON.csv",
        "PIPELINE_FAMILY_RESULTS.csv",
        "COST_MODEL_COMPARISON.csv",
        "COST_SENSITIVITY.csv",
        "BOOTSTRAP_INTERVALS.csv",
        "HOLM_CORRECTION.csv",
        "SELECTION_STABILITY.csv",
        "H10_C4_FINAL_STATUS.json",
        "DEVELOPMENT_COST_CALIBRATION.json",
    )
    missing = [
        name for name in required if not (root / "results/h10_c4" / name).exists()
    ]
    with (root / "results/h10_c4/SCENARIO_MANIFEST.csv").open(
        encoding="utf-8"
    ) as handle:
        scenarios = list(csv.DictReader(handle))
    with (root / "results/h10_c4/PLAN_LEVEL_RESULTS.csv").open(
        encoding="utf-8"
    ) as handle:
        plans = list(csv.DictReader(handle))
    checks = {
        "required_outputs": not missing,
        "held_out_scenario_count": len(scenarios) == 120,
        "pipeline_family_count": len({row["pipeline_family"] for row in scenarios}) == 6,
        "strategies_per_scenario": len(plans) == 480,
        "twenty_scenarios_per_family": all(
            sum(row["pipeline_family"] == family for row in scenarios) == 20
            for family in {row["pipeline_family"] for row in scenarios}
        ),
        "ten_mutation_families": (
            len({row["mutation_family"] for row in scenarios}) == 10
        ),
        "controlled_labels_only": all(
            row["scenario_type"] == SCENARIO_LABEL for row in scenarios
        ),
    }
    return {
        "checks": checks,
        "missing": missing,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
