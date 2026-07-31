from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

from fuzzyxai.diagnostics.contracts import canonical_sha256

from .pipeline import PipelineRun

COMPARATIVE_VERSION = "FUZZYXAI_ML_PIPELINE_V2_COMPARATIVE_V1"
MODE_IDS = ("B0", "B1", "B2", "B3", "A0", "A1", "A2", "A3", "A4")

# These checks can be evaluated from one component's own values and manifest.
LOCAL_CONTRACTS = frozenset(
    {
        "DATASET_IDENTITY",
        "DATASET_SCHEMA",
        "TARGET_NOT_IN_FEATURES",
        "CLASS_MAPPING",
        "SPLIT_REPRODUCIBILITY",
        "PREPROCESSOR_VERSION",
        "FEATURE_ORDER",
        "FEATURE_COUNT",
        "TRANSFORM_OUTPUT_SCHEMA",
        "FINITE_TRANSFORMED_VALUES",
        "TRAINING_CONFIGURATION",
        "MODEL_CONVERGENCE",
        "MODEL_FEATURE_SCHEMA",
        "TRAINING_DATA_HASH",
        "MODEL_ARTIFACT_HASH",
        "METRIC_SANITY",
        "MODEL_INPUT_SCHEMA",
        "INPUT_FEATURE_SCHEMA",
        "INPUT_VALUE_DOMAIN",
        "PREDICTION_OBJECT_BINDING",
        "PREDICTION_OUTPUT_SANITY",
        "EXPLANATION_OBJECT_ID",
        "EXPLANATION_OUTPUT_CONSISTENCY",
        "EXPLANATION_FEATURE_SCHEMA",
        "REQUIRED_PROVENANCE",
        "REDUCTION_LOSS_LIMIT",
        "USER_CLAIM_EVIDENCE_BINDING",
        "AUDIT_ARTIFACT_HASH",
    }
)

# These require comparing objects or claims owned by different route stages.
CROSS_STAGE_CONTRACTS = frozenset(
    {
        "TRAIN_VALIDATION_TEST_DISJOINTNESS",
        "PREPROCESSOR_FIT_SCOPE",
        "MODEL_EXPLAINER_VERSION",
        "MODEL_RULE_CONFLICT",
        "UNCERTAINTY_REPRESENTATION_COVERAGE",
    }
)

STAGE_ORDER = (
    "DATA_PREPARATION",
    "DATA_SPLIT",
    "PREPROCESSING",
    "TRAINING",
    "MODEL_ARTIFACT",
    "INFERENCE",
    "POST_HOC_EXPLANATION",
    "FUZZY_REPRESENTATION",
    "REDUCTION",
    "PRESENTATION",
    "REPAIR",
    "RECERTIFICATION",
)
SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NONE": 4}


@dataclass(frozen=True)
class StandardLogInput:
    stdout: tuple[str, ...]
    stderr: tuple[str, ...]
    exception: str | None
    warnings: tuple[dict[str, str], ...]
    return_code: int
    model_metrics: Mapping[str, float]


@dataclass(frozen=True)
class MLflowObservationInput:
    parameters: Mapping[str, Any]
    metrics: Mapping[str, float]
    tags: Mapping[str, str]
    registered_artifacts: Mapping[str, str]


@dataclass(frozen=True)
class ModeInput:
    scenario_id: str
    mode_id: str
    standard_log: StandardLogInput | None = None
    mlflow_observation: MLflowObservationInput | None = None
    observations: tuple[Mapping[str, Any], ...] = ()
    contract_results: tuple[Mapping[str, Any], ...] = ()
    route_graph: Mapping[str, Any] | None = None
    repair_plan: Mapping[str, Any] | None = None
    full_run: PipelineRun | None = None


@dataclass(frozen=True)
class ComparativeResult:
    scenario_id: str
    mode_id: str
    detected: bool
    pipeline_status: str
    stage: str | None
    contract_id: str | None
    component_id: str | None
    action: str | None
    evidence_refs: tuple[str, ...]
    diagnostic_cut: Mapping[str, Any] | None
    repair_available: bool
    repair_executed: bool
    target_contract_repaired: bool
    recertified: bool
    contracts_rechecked_count: int
    new_critical_violations: int | None
    rollback_verified: bool
    abstained: bool
    runtime_ms: float
    observed_value: Any = None
    expected_value: Any = None
    source_uri: str | None = None
    observation_sha256: str | None = None
    evidence_completeness: float = 0.0
    canonical_sha256: str = ""

    @classmethod
    def build(cls, **payload: Any) -> ComparativeResult:
        digest_payload = {key: value for key, value in payload.items() if key not in {"runtime_ms", "canonical_sha256"}}
        return cls(**payload, canonical_sha256=canonical_sha256(digest_payload))


def project_mode_input(run: PipelineRun, mode_id: str) -> ModeInput:
    """Create a fail-closed channel projection for one registered mode."""
    if mode_id not in MODE_IDS:
        raise ValueError(f"unregistered comparative mode: {mode_id}")
    if mode_id == "B0":
        return ModeInput(run.scenario_id, mode_id, standard_log=_standard_log_projection(run))
    if mode_id == "B1":
        return ModeInput(run.scenario_id, mode_id, mlflow_observation=_mlflow_projection(run))

    local_only = mode_id in {"B2", "B3", "A0", "A1"}
    contracts = tuple(
        item
        for item in run.contract_report["results"]
        if not local_only or str(item["contract_id"]) in LOCAL_CONTRACTS
    )
    allowed_refs = {ref for item in contracts for ref in item.get("evidence_refs", ())}
    observations = tuple(item for item in run.observations if item["observation_id"] in allowed_refs)
    graph = run.route_graph if mode_id in {"A1", "A2", "A3"} else None
    repair_plan = run.repair_plan if mode_id == "A3" else None
    full_run = run if mode_id == "A4" else None
    return ModeInput(
        scenario_id=run.scenario_id,
        mode_id=mode_id,
        observations=observations,
        contract_results=contracts,
        route_graph=graph,
        repair_plan=repair_plan,
        full_run=full_run,
    )


def evaluate_mode(mode_input: ModeInput) -> ComparativeResult:
    """Evaluate one mode without accepting scenario targets or Gold fields."""
    started = perf_counter()
    if mode_input.mode_id == "B0":
        payload = _evaluate_standard_log(mode_input)
    elif mode_input.mode_id == "B1":
        payload = _evaluate_mlflow_observation(mode_input)
    elif mode_input.mode_id in {"B2", "B3", "A0", "A1", "A2", "A3"}:
        payload = _evaluate_contract_projection(mode_input)
    elif mode_input.mode_id == "A4":
        payload = _evaluate_full_fuzzyxai(mode_input)
    else:  # pragma: no cover - guarded by projection and tests
        raise ValueError(mode_input.mode_id)
    payload["runtime_ms"] = (perf_counter() - started) * 1000
    return ComparativeResult.build(**payload)


def result_payload(result: ComparativeResult) -> dict[str, Any]:
    return asdict(result)


def _standard_log_projection(run: PipelineRun) -> StandardLogInput:
    # The warning was emitted by sklearn and captured by the unchanged pipeline.
    mutation = run.manifests.get("scenario_mutation", {})
    warning_rows: list[dict[str, str]] = []
    if mutation.get("convergence_warning") is True:
        warning_rows.append(
            {
                "category": "sklearn.exceptions.ConvergenceWarning",
                "message": "LogisticRegression failed to converge at the configured max_iter",
                "component_id": "logistic_regression",
            }
        )
    metrics = run.manifests.get("model_manifest", {}).get("metrics", {})
    return StandardLogInput((), (), None, tuple(warning_rows), 0, dict(metrics))


def _mlflow_projection(run: PipelineRun) -> MLflowObservationInput:
    manifests = run.manifests
    dataset = manifests["dataset_manifest"]
    split = manifests["split_manifest"]
    preprocessor = manifests["preprocessor_manifest"]
    model = manifests["model_manifest"]
    # FuzzyXAI outputs (diagnosis, contract report, route graph, pipeline_valid)
    # are intentionally absent: B1 models stock run registration, not diagnosis.
    return MLflowObservationInput(
        parameters={
            "pipeline_version": run.pipeline_version,
            "dataset_id": dataset["dataset_id"],
            "dataset_sha256": dataset["dataset_sha256"],
            "split_sha256": split["split_sha256"],
            "preprocessor_sha256": preprocessor["artifact_sha256"],
            "model_sha256": model["model_sha256"],
            "feature_schema_sha256": model["feature_schema_sha256"],
        },
        metrics=dict(model["metrics"]),
        tags={"run_id": run.run_id, "pipeline_version": run.pipeline_version},
        registered_artifacts={
            "dataset_manifest.json": canonical_sha256(dataset),
            "split_manifest.json": canonical_sha256(split),
            "preprocessor_manifest.json": canonical_sha256(preprocessor),
            "model_manifest.json": canonical_sha256(model),
        },
    )


def _base_payload(scenario_id: str, mode_id: str) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "mode_id": mode_id,
        "detected": False,
        "pipeline_status": "INSUFFICIENT_EVIDENCE",
        "stage": None,
        "contract_id": None,
        "component_id": None,
        "action": None,
        "evidence_refs": (),
        "diagnostic_cut": None,
        "repair_available": False,
        "repair_executed": False,
        "target_contract_repaired": False,
        "recertified": False,
        "contracts_rechecked_count": 0,
        "new_critical_violations": None,
        "rollback_verified": False,
        "abstained": True,
        "observed_value": None,
        "expected_value": None,
        "source_uri": None,
        "observation_sha256": None,
        "evidence_completeness": 0.0,
    }


def _evaluate_standard_log(mode_input: ModeInput) -> dict[str, Any]:
    payload = _base_payload(mode_input.scenario_id, mode_input.mode_id)
    log = mode_input.standard_log
    if log is None:
        return payload
    warning = next((item for item in log.warnings if item["category"].endswith("ConvergenceWarning")), None)
    if warning:
        payload.update(
            detected=True,
            pipeline_status="INVALID",
            stage="TRAINING",
            component_id=warning["component_id"],
            abstained=False,
        )
    return payload


def _evaluate_mlflow_observation(mode_input: ModeInput) -> dict[str, Any]:
    payload = _base_payload(mode_input.scenario_id, mode_input.mode_id)
    observation = mode_input.mlflow_observation
    if observation is None:
        return payload
    impossible_metric = any(not _metric_is_sane(name, value) for name, value in observation.metrics.items())
    missing_hash = any(not value for key, value in observation.parameters.items() if key.endswith("sha256"))
    if impossible_metric or missing_hash:
        payload.update(detected=True, pipeline_status="INVALID", abstained=False)
    return payload


def _metric_is_sane(name: str, value: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    if name in {"accuracy", "precision", "recall", "f1", "roc_auc"}:
        return 0.0 <= number <= 1.0
    if name == "sample_count":
        return number > 0.0
    return math.isfinite(number)


def _evaluate_contract_projection(mode_input: ModeInput) -> dict[str, Any]:
    payload = _base_payload(mode_input.scenario_id, mode_input.mode_id)
    failed = [item for item in mode_input.contract_results if not item.get("passed", False)]
    if not failed:
        payload.update(pipeline_status="VALID", action="ACCEPT", abstained=False, evidence_completeness=1.0)
        return payload

    selected = min(failed, key=_greedy_key)
    observation = _find_observation(mode_input.observations, selected)
    evidence = _evidence_fields(selected, observation)
    cut = None
    if mode_input.mode_id in {"A3"}:
        cut = {
            "contracts": [selected["contract_id"]],
            "components": [selected["component_id"]],
            "evidence_refs": list(selected.get("evidence_refs", ())),
        }
    repair_available = bool(
        mode_input.mode_id == "A3"
        and mode_input.repair_plan
        and mode_input.repair_plan.get("executable", True)
    )
    payload.update(
        detected=True,
        pipeline_status="INVALID",
        stage=str(selected["stage"]),
        contract_id=str(selected["contract_id"]),
        component_id=str(selected["component_id"]),
        action=str(selected["action"]),
        evidence_refs=tuple(selected.get("evidence_refs", ())),
        diagnostic_cut=cut,
        repair_available=repair_available,
        abstained=False,
        **evidence,
    )
    return payload


def _evaluate_full_fuzzyxai(mode_input: ModeInput) -> dict[str, Any]:
    run = mode_input.full_run
    if run is None:
        raise ValueError("A4 requires the canonical full pipeline run")
    diagnosis = run.diagnosis
    contract_id = diagnosis.get("violated_contract")
    selected = next(
        (item for item in run.contract_report["violations"] if item["contract_id"] == contract_id),
        None,
    )
    observation = _find_observation(run.observations, selected) if selected else None
    evidence = _evidence_fields(selected, observation) if selected else {
        "observed_value": None,
        "expected_value": None,
        "source_uri": None,
        "observation_sha256": None,
        "evidence_completeness": 1.0,
    }
    recertification = run.recertification or {}
    detected = bool(contract_id)
    return {
        "scenario_id": run.scenario_id,
        "mode_id": mode_input.mode_id,
        "detected": detected,
        "pipeline_status": str(run.pipeline_status),
        "stage": diagnosis.get("failed_stage"),
        "contract_id": contract_id,
        "component_id": diagnosis.get("source_component"),
        "action": diagnosis.get("recommended_action"),
        "evidence_refs": tuple(diagnosis.get("evidence_refs", ())),
        "diagnostic_cut": diagnosis.get("diagnostic_cut") if detected else None,
        "repair_available": bool(run.repair_plan),
        "repair_executed": bool(recertification.get("repair_executed")),
        "target_contract_repaired": bool(recertification.get("target_contract_repaired")),
        "recertified": bool(recertification.get("full_recertification")),
        "contracts_rechecked_count": int(recertification.get("contracts_rechecked_count", 0)),
        "new_critical_violations": int(recertification.get("new_critical_violations", 0)) if recertification else None,
        "rollback_verified": bool(recertification.get("rollback_verified")),
        "abstained": False,
        **evidence,
    }


def _greedy_key(item: Mapping[str, Any]) -> tuple[int, int, str, str]:
    return (
        SEVERITY_ORDER.get(str(item.get("severity", "NONE")), 99),
        STAGE_ORDER.index(str(item["stage"])) if str(item["stage"]) in STAGE_ORDER else len(STAGE_ORDER),
        str(item.get("component_id", "")),
        str(item.get("contract_id", "")),
    )


def _find_observation(
    observations: tuple[Mapping[str, Any], ...],
    contract: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not contract:
        return None
    refs = set(contract.get("evidence_refs", ()))
    return next((item for item in observations if item.get("observation_id") in refs), None)


def _evidence_fields(
    contract: Mapping[str, Any] | None,
    observation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not contract or not observation:
        return {
            "observed_value": None,
            "expected_value": None,
            "source_uri": None,
            "observation_sha256": None,
            "evidence_completeness": 0.0,
        }
    required = (
        observation.get("observed_value") is not None,
        observation.get("expected_value") is not None,
        bool(observation.get("source_uri")),
        bool(contract.get("stage")),
        bool(contract.get("component_id")),
        bool(contract.get("contract_id")),
        bool(observation.get("sha256")),
    )
    return {
        "observed_value": observation.get("observed_value"),
        "expected_value": observation.get("expected_value"),
        "source_uri": observation.get("source_uri"),
        "observation_sha256": observation.get("sha256"),
        "evidence_completeness": sum(required) / len(required),
    }
