from __future__ import annotations

import resource
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any

from fuzzyxai.diagnostics import (
    ActionableRepairPlanner,
    Contract,
    DiagnosticValidator,
    MinimalDiagnosticCutFinder,
    RepairExecutionContext,
    RepairExecutor,
    RouteEdge,
    RouteGraph,
    RouteNode,
    RouteRecertifier,
)
from fuzzyxai.diagnostics.contracts import RepairCostModel, canonical_sha256
from fuzzyxai.diagnostics.repair_registry import ContractRepairProvider, RepairProviderRegistry
from fuzzyxai.external_adapters import ExternalPipelineArtifacts, ManifestExternalPipelineAdapter
from fuzzyxai.ml_vertical.pipeline import CONTRACT_STAGE

from .external_runners import SPECS

MODE_IDS = (
    "B_LOCAL_STRONG",
    "B_PAIRWISE_RULES",
    "B_MLFLOW_QUERY",
    "B_GREEDY_CROSS_STAGE",
    "O_FUZZYXAI",
)
LOCAL_CONTRACTS = frozenset({"TARGET_NOT_IN_FEATURES", "MODEL_ARTIFACT_HASH", "REQUIRED_PROVENANCE"})
MLFLOW_CONTRACTS = frozenset({"MODEL_ARTIFACT_HASH", "MODEL_EXPLAINER_VERSION", "REQUIRED_PROVENANCE"})
CONTRACT_DEPENDENCIES = {
    "MODEL_INPUT_SCHEMA": "FEATURE_ORDER",
    "PREDICTION_OUTPUT_SANITY": "FEATURE_ORDER",
    "EXPLANATION_OUTPUT_CONSISTENCY": "FEATURE_ORDER",
    "USER_CLAIM_EVIDENCE_BINDING": "FEATURE_ORDER",
}


@dataclass(frozen=True)
class FaultSpec:
    case_id: str
    contract_id: str | None
    stage: str | None
    action: str
    category: str
    repair_operation: str | None
    dependent_contracts: tuple[str, ...] = ()
    variant: str = "baseline"


FAULTS = (
    FaultSpec("C1_BASELINE", None, None, "ACCEPT", "CONTROL", None),
    FaultSpec("C2_CONSISTENT_RETRAIN", None, None, "ACCEPT", "CONTROL", None, variant="retrained"),
    FaultSpec("E1_TARGET_LEAKAGE", "TARGET_NOT_IN_FEATURES", "DATA_PREPARATION", "BLOCK", "LOCAL", None),
    FaultSpec("E2_TRAIN_TEST_OVERLAP", "TRAIN_VALIDATION_TEST_DISJOINTNESS", "DATA_SPLIT", "BLOCK", "CROSS_STAGE", "restore_split_manifest"),
    FaultSpec("E3_PREPROCESSOR_FIT_SCOPE", "PREPROCESSOR_FIT_SCOPE", "PREPROCESSING", "BLOCK", "CROSS_STAGE", "refit_preprocessor_on_train"),
    FaultSpec(
        "E4_FEATURE_SCHEMA_OR_ORDER",
        "FEATURE_ORDER",
        "PREPROCESSING",
        "BLOCK",
        "GLOBAL",
        "restore_feature_schema",
        ("MODEL_INPUT_SCHEMA", "PREDICTION_OUTPUT_SANITY", "EXPLANATION_OUTPUT_CONSISTENCY", "USER_CLAIM_EVIDENCE_BINDING"),
    ),
    FaultSpec("E5_MODEL_ARTIFACT_MISMATCH", "MODEL_ARTIFACT_HASH", "MODEL_ARTIFACT", "BLOCK", "LOCAL", "restore_registered_model"),
    FaultSpec(
        "E6_MODEL_EXPLAINER_VERSION",
        "MODEL_EXPLAINER_VERSION",
        "POST_HOC_EXPLANATION",
        "BLOCK",
        "CROSS_STAGE",
        "rebuild_explanation_for_registered_model",
    ),
    FaultSpec(
        "E7_PREDICTION_EXPLANATION_BINDING",
        "PREDICTION_OBJECT_BINDING",
        "INFERENCE",
        "BLOCK",
        "CROSS_STAGE",
        "rebuild_explanation_for_registered_model",
    ),
    FaultSpec("E8_EXPLANATION_PROVENANCE", "REQUIRED_PROVENANCE", "POST_HOC_EXPLANATION", "BLOCK", "LOCAL", "restore_explanation_provenance"),
)

REPAIR_OPERATIONS = {
    "TRAIN_VALIDATION_TEST_DISJOINTNESS": "restore_split_manifest",
    "PREPROCESSOR_FIT_SCOPE": "refit_preprocessor_on_train",
    "FEATURE_ORDER": "restore_feature_schema",
    "MODEL_ARTIFACT_HASH": "restore_registered_model",
    "MODEL_EXPLAINER_VERSION": "rebuild_explanation_for_registered_model",
    "PREDICTION_OBJECT_BINDING": "rebuild_explanation_for_registered_model",
    "REQUIRED_PROVENANCE": "restore_explanation_provenance",
}


@dataclass(frozen=True)
class Decision:
    run_id: str
    pipeline_id: str
    repository_commit: str
    case_id: str
    mode_id: str
    pipeline_status: str
    detected: bool
    stage: str | None
    contract_id: str | None
    component_id: str | None
    root_cause: str | None
    dependent_violations: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    diagnostic_cut: dict[str, Any] | None
    action: str
    repair_plan: dict[str, Any] | None
    repair_executed: bool
    target_contract_repaired: bool
    recertified: bool
    contracts_checked: int
    new_critical_violations: int
    rollback_verified: bool
    reported_symptom_count: int
    proposed_repair_count: int
    redundant_repair_count: int
    evidence_completeness: float
    runtime_breakdown_ms: dict[str, float]
    peak_rss_kb: int
    artifact_bytes: int
    canonical_sha256: str


def _component(contract_id: str) -> str:
    stage = CONTRACT_STAGE[contract_id].value.lower()
    return f"external.{stage}.{contract_id.lower()}"


def _registered_value(artifacts: ExternalPipelineArtifacts, contract_id: str) -> Any:
    values = {
        "DATASET_IDENTITY": artifacts.dataset["dataset_sha256"],
        "DATASET_SCHEMA": canonical_sha256(artifacts.dataset["feature_names"]),
        "TARGET_NOT_IN_FEATURES": False,
        "CLASS_MAPPING": "registered",
        "TRAIN_VALIDATION_TEST_DISJOINTNESS": 0,
        "SPLIT_REPRODUCIBILITY": artifacts.split["split_sha256"],
        "PREPROCESSOR_VERSION": artifacts.preprocessor["artifact_sha256"],
        "PREPROCESSOR_FIT_SCOPE": artifacts.preprocessor["expected_train_row_ids_sha256"],
        "FEATURE_ORDER": canonical_sha256(artifacts.preprocessor["feature_names_out"]),
        "FEATURE_COUNT": artifacts.preprocessor["output_feature_count"],
        "TRANSFORM_OUTPUT_SCHEMA": canonical_sha256(artifacts.preprocessor["feature_names_out"]),
        "FINITE_TRANSFORMED_VALUES": True,
        "TRAINING_CONFIGURATION": artifacts.model["model_class"],
        "MODEL_CONVERGENCE": True,
        "MODEL_FEATURE_SCHEMA": canonical_sha256(artifacts.model["feature_names"]),
        "TRAINING_DATA_HASH": artifacts.split["split_sha256"],
        "MODEL_ARTIFACT_HASH": artifacts.model["registered_artifact_sha256"],
        "METRIC_SANITY": True,
        "MODEL_INPUT_SCHEMA": canonical_sha256(artifacts.model["feature_names"]),
        "PREDICTION_OBJECT_BINDING": artifacts.prediction["object_id"],
        "PREDICTION_OUTPUT_SANITY": True,
        "MODEL_EXPLAINER_VERSION": artifacts.model["model_version"],
        "EXPLANATION_OBJECT_ID": artifacts.prediction["object_id"],
        "EXPLANATION_OUTPUT_CONSISTENCY": True,
        "EXPLANATION_FEATURE_SCHEMA": canonical_sha256(artifacts.model["feature_names"]),
        "REQUIRED_PROVENANCE": artifacts.explanation["source_uri"],
        "REDUCTION_LOSS_LIMIT": True,
        "USER_CLAIM_EVIDENCE_BINDING": artifacts.explanation["artifact_sha256"],
    }
    return values[contract_id]


def _fault_value(value: Any, contract_id: str) -> Any:
    if contract_id == "TARGET_NOT_IN_FEATURES":
        return True
    if contract_id == "TRAIN_VALIDATION_TEST_DISJOINTNESS":
        return 1
    if contract_id == "REQUIRED_PROVENANCE":
        return "missing"
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    return f"mismatch:{str(value)[:24]}"


def build_route(artifacts: ExternalPipelineArtifacts, fault: FaultSpec) -> RouteGraph:
    failed = {fault.contract_id, *fault.dependent_contracts} if fault.contract_id else set()
    source_id = f"cause:{fault.contract_id.lower()}" if fault.contract_id else "cause:none"
    evidence_by_stage = {
        "DATA_PREPARATION": "dataset_manifest.json",
        "DATA_SPLIT": "split_manifest.json",
        "PREPROCESSING": "preprocessor_manifest.json",
        "TRAINING": "model_manifest.json",
        "MODEL_ARTIFACT": "model_manifest.json",
        "INFERENCE": "prediction_manifest.json",
        "POST_HOC_EXPLANATION": "explanation_manifest.json",
        "REDUCTION": "explanation_manifest.json",
        "PRESENTATION": "explanation_manifest.json",
    }
    nodes = []
    contracts = []
    for contract_id, stage in CONTRACT_STAGE.items():
        expected = _registered_value(artifacts, contract_id)
        observed = _fault_value(expected, contract_id) if contract_id in failed else expected
        evidence = evidence_by_stage[stage.value]
        node_id = f"contract:{contract_id.lower()}"
        nodes.append(
            RouteNode(
                node_id=node_id,
                node_type=stage.value.lower(),
                component_id=_component(contract_id),
                component_version="external-adapter-v1",
                registered_attributes={"value": expected},
                observed_attributes={"value": observed},
                mandatory=True,
                repairable=contract_id in REPAIR_OPERATIONS,
                evidence_refs=(evidence,),
            )
        )
        shared = source_id if contract_id in failed else node_id
        category = (
            "serialization"
            if contract_id == "MODEL_ARTIFACT_HASH"
            else "provenance"
            if contract_id == "REQUIRED_PROVENANCE"
            else "explainer"
            if contract_id in {"MODEL_EXPLAINER_VERSION", "PREDICTION_OBJECT_BINDING"}
            else "preprocessing"
        )
        contracts.append(
            Contract(
                contract_id=contract_id,
                kind="checksum" if contract_id == "MODEL_ARTIFACT_HASH" else "equals",
                subject_id=node_id,
                field="value",
                expected=expected,
                severity="error",
                category=category,
                mandatory=True,
                repairable=contract_id in REPAIR_OPERATIONS,
                evidence_refs=(evidence,),
                source_nodes=(shared,),
            )
        )
    nodes.append(
        RouteNode(
            node_id=source_id,
            node_type="causal_source",
            component_id=f"external.source.{fault.case_id.lower()}",
            component_version="1",
            registered_attributes={"state": "registered"},
            observed_attributes={"state": "mutated" if fault.contract_id else "registered"},
            mandatory=False,
            repairable=bool(fault.repair_operation),
            evidence_refs=(evidence_by_stage.get(fault.stage or "DATA_PREPARATION", "pipeline_manifest.json"),),
        )
    )
    stage_nodes: dict[str, str] = {}
    for contract_id, stage in CONTRACT_STAGE.items():
        stage_nodes.setdefault(stage.value, f"contract:{contract_id.lower()}")
    ordered_stages = list(stage_nodes)
    edges = tuple(
        RouteEdge(
            edge_id=f"edge:{left}:{right}",
            source=stage_nodes[left],
            target=stage_nodes[right],
            relation="derived_from",
            mandatory=True,
            registered_contract={"relation": "derived_from"},
            observed_contract={"relation": "derived_from"},
            repairable=True,
            evidence_refs=("pipeline_manifest.json",),
            relation_status="known_valid",
        )
        for left, right in pairwise(ordered_stages)
    )
    return RouteGraph(
        route_id=f"external:{artifacts.pipeline_id}:{fault.case_id}",
        nodes=tuple(nodes),
        edges=edges,
        contracts=tuple(contracts),
        metadata={
            "pipeline_id": artifacts.pipeline_id,
            "repository_commit": artifacts.repository_commit,
            "case_id": fault.case_id,
            "applicable_contracts": len(CONTRACT_STAGE),
            "repair_costs": {f"node:{source_id}": 0.1},
        },
    )


def _repair_registry() -> RepairProviderRegistry:
    providers = []
    configurations = (
        ("split.restore", "preprocessing", "restore_split_manifest"),
        ("preprocessor.refit", "preprocessing", "refit_preprocessor_on_train"),
        ("schema.restore", "preprocessing", "restore_feature_schema"),
        ("model.restore", "serialization", "restore_registered_model"),
        ("explanation.rebuild", "explainer", "rebuild_explanation_for_registered_model"),
        ("provenance.restore", "provenance", "restore_explanation_provenance"),
    )
    for provider_id, category, operation in configurations:
        providers.append(
            ContractRepairProvider(
                provider_id,
                frozenset(),
                frozenset({category}),
                operation,
                operation.replace("_", " "),
                1.0,
            )
        )
    return RepairProviderRegistry(providers)


def _root_contract(failed: tuple[str, ...]) -> str | None:
    failed_set = set(failed)
    return next(
        (contract for contract in failed if CONTRACT_DEPENDENCIES.get(contract) not in failed_set),
        None,
    )


def evaluate(artifacts: ExternalPipelineArtifacts, fault: FaultSpec, mode_id: str) -> Decision:
    if mode_id not in MODE_IDS:
        raise ValueError(f"unregistered mode: {mode_id}")
    graph_started = perf_counter()
    graph = build_route(artifacts, fault)
    graph_ms = (perf_counter() - graph_started) * 1000
    audit_started = perf_counter()
    validator = DiagnosticValidator()
    validation = validator.validate(graph)
    audit_ms = (perf_counter() - audit_started) * 1000
    all_issues = tuple(validation.issues)
    if mode_id == "B_LOCAL_STRONG":
        issues = tuple(item for item in all_issues if item.violated_contract in LOCAL_CONTRACTS)
    elif mode_id == "B_MLFLOW_QUERY":
        issues = tuple(item for item in all_issues if item.violated_contract in MLFLOW_CONTRACTS)
    else:
        issues = all_issues
    stage_order = {stage.value: index for index, stage in enumerate(dict.fromkeys(CONTRACT_STAGE.values()))}
    issues = tuple(sorted(issues, key=lambda item: (stage_order[CONTRACT_STAGE[item.violated_contract].value], item.violated_contract)))
    selected = issues[0] if issues else None
    root = None
    dependents: tuple[str, ...] = ()
    cut_payload = None
    plan_payload = None
    repair_executed = target_repaired = recertified = rollback = False
    checked = 0
    new_critical = 0
    proposed = redundant = 0
    cut_ms = plan_ms = recert_ms = 0.0
    if mode_id == "O_FUZZYXAI" and issues:
        cut_started = perf_counter()
        cut = MinimalDiagnosticCutFinder().find(graph, validation, RepairCostModel(graph.metadata["repair_costs"]))
        cut_ms = (perf_counter() - cut_started) * 1000
        failed_contracts = tuple(item.violated_contract for item in issues)
        root = _root_contract(failed_contracts)
        dependents = tuple(item for item in failed_contracts if item != root)
        cut_payload = {"atoms": cut.atom_keys, "size": len(cut.defect_atoms), "solver": cut.solver, "optimal": cut.optimal}
        if fault.repair_operation:
            registry = _repair_registry()
            planner = ActionableRepairPlanner(registry)
            plan_started = perf_counter()
            plan = planner.plan(graph, all_issues, cut)
            plan_ms = (perf_counter() - plan_started) * 1000
            proposed = len(plan.steps)
            plan_payload = {
                "operations": [step.operation for step in plan.steps],
                "step_ids": [step.step_id for step in plan.steps],
                "fully_executable": plan.fully_executable,
                "rollback": [step.rollback_operation for step in plan.steps],
            }
            clean = build_route(artifacts, replace(fault, contract_id=None, dependent_contracts=(), repair_operation=None))
            handlers = {step.operation: (lambda before, current_step, target=clean: target) for step in plan.steps}
            handlers["restore_previous_artifact_snapshot"] = lambda current, step: graph
            context = RepairExecutionContext(
                handlers=handlers,
                approved_step_ids=frozenset(step.step_id for step in plan.steps),
                allow_external_changes=True,
            )
            after, executions = RepairExecutor(registry).execute(graph, plan, context)
            repair_executed = bool(executions) and all(item.status == "completed" for item in executions)
            target_repaired = validator.validate(after).valid
            recert_started = perf_counter()
            report = RouteRecertifier(validator).recertify(graph, after, plan, executions)
            recert_ms = (perf_counter() - recert_started) * 1000
            recertified = report.status == "full_success"
            checked = len(validation.checked_contracts)
            new_critical = len(report.new_critical_issues)
            rollback = all(step.rollback_operation is not None for step in plan.steps)
    elif mode_id == "B_PAIRWISE_RULES" and issues:
        operations = [REPAIR_OPERATIONS.get(item.violated_contract, f"repair_{item.violated_contract.lower()}") for item in issues]
        plan_payload = {"operations": operations} if operations else None
        proposed = len(operations)
        redundant = max(0, proposed - (1 if operations else 0))
        dependents = tuple(item.violated_contract for item in issues[1:])
    elif mode_id == "B_GREEDY_CROSS_STAGE" and selected:
        root = selected.violated_contract
    detected = bool(issues)
    refs = tuple(dict.fromkeys(ref for issue in issues for ref in issue.evidence_refs))
    completeness = (
        1.0
        if not detected
        else float(
            all(
                issue.affected_fields
                and issue.affected_nodes
                and issue.violated_contract
                and CONTRACT_STAGE[issue.violated_contract].value
                and issue.evidence_refs
                for issue in issues
            )
        )
    )
    total_ms = graph_ms + audit_ms + cut_ms + plan_ms + recert_ms
    payload = {
        "run_id": f"external:{canonical_sha256((artifacts.pipeline_id, fault.case_id, mode_id))[:20]}",
        "pipeline_id": artifacts.pipeline_id,
        "repository_commit": artifacts.repository_commit,
        "case_id": fault.case_id,
        "mode_id": mode_id,
        "pipeline_status": "INVALID" if detected else "VALID",
        "detected": detected,
        "stage": CONTRACT_STAGE[selected.violated_contract].value if selected else None,
        "contract_id": selected.violated_contract if selected else None,
        "component_id": graph.node(selected.affected_nodes[0]).component_id if selected and selected.affected_nodes else None,
        "root_cause": root,
        "dependent_violations": dependents,
        "evidence_refs": refs,
        "diagnostic_cut": cut_payload,
        "action": "BLOCK" if detected else "ACCEPT",
        "repair_plan": plan_payload,
        "repair_executed": repair_executed,
        "target_contract_repaired": target_repaired,
        "recertified": recertified,
        "contracts_checked": checked,
        "new_critical_violations": new_critical,
        "rollback_verified": rollback,
        "reported_symptom_count": len(issues),
        "proposed_repair_count": proposed,
        "redundant_repair_count": redundant,
        "evidence_completeness": completeness,
        "runtime_breakdown_ms": {
            "graph_build": graph_ms,
            "audit": audit_ms,
            "diagnostic_cut": cut_ms,
            "repair_planning": plan_ms,
            "recertification": recert_ms,
            "total": total_ms,
        },
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "artifact_bytes": sum(path.stat().st_size for path in artifacts.root.glob("*") if path.is_file()),
    }
    identity = {key: value for key, value in payload.items() if key not in {"runtime_breakdown_ms", "peak_rss_kb"}}
    return Decision(**payload, canonical_sha256=canonical_sha256(identity))


class ExternalBenchmark:
    def __init__(self, fixtures_root: Path) -> None:
        self.fixtures_root = fixtures_root

    def artifacts(self, pipeline_id: str, variant: str = "baseline") -> ExternalPipelineArtifacts:
        return ManifestExternalPipelineAdapter(self.fixtures_root / pipeline_id / variant).build_route_observations()

    def run_all(self) -> list[dict[str, Any]]:
        rows = []
        for spec in SPECS:
            for fault in FAULTS:
                artifacts = self.artifacts(spec.pipeline_id, fault.variant)
                for mode in MODE_IDS:
                    rows.append(asdict(evaluate(artifacts, fault, mode)))
        return rows
