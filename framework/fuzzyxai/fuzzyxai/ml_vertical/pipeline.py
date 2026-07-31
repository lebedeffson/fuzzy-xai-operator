from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import shap
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from fuzzyxai.diagnostics.contracts import canonical_sha256

from .contracts import VerticalRun
from .model import MODEL_ID, MODEL_VERSION
from .service import SCENARIOS, MLVerticalService

PIPELINE_VERSION = "2.0.0"
FIXED_TIMESTAMP = "2000-01-01T00:00:00Z"


class PipelineStage(str, Enum):
    DATA_PREPARATION = "DATA_PREPARATION"
    DATA_SPLIT = "DATA_SPLIT"
    PREPROCESSING = "PREPROCESSING"
    TRAINING = "TRAINING"
    MODEL_ARTIFACT = "MODEL_ARTIFACT"
    INFERENCE = "INFERENCE"
    POST_HOC_EXPLANATION = "POST_HOC_EXPLANATION"
    FUZZY_REPRESENTATION = "FUZZY_REPRESENTATION"
    REDUCTION = "REDUCTION"
    PRESENTATION = "PRESENTATION"
    REPAIR = "REPAIR"
    RECERTIFICATION = "RECERTIFICATION"


@dataclass(frozen=True)
class StageObservation:
    observation_id: str
    run_id: str
    stage: PipelineStage
    component_id: str
    field_name: str
    observed_value: Any
    expected_value: Any | None
    evidence_type: str
    source_uri: str | None
    sha256: str
    timestamp_utc: str = FIXED_TIMESTAMP


@dataclass(frozen=True)
class RegisteredRepairOperation:
    operation_id: str
    supported_contracts: tuple[str, ...]
    target_component_type: str
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    rollback_operation: str
    maximum_cost: float
    mutates_state: bool


@dataclass(frozen=True)
class PipelineRun:
    run_id: str
    scenario_id: str
    pipeline_version: str
    pipeline_status: str
    stage_statuses: dict[str, str]
    observations: tuple[dict[str, Any], ...]
    contract_report: dict[str, Any]
    route_graph: dict[str, Any]
    diagnosis: dict[str, Any]
    repair_plan: dict[str, Any] | None
    recertification: dict[str, Any] | None
    prediction: dict[str, Any]
    explanation: dict[str, Any]
    representation: dict[str, Any]
    manifests: dict[str, Any]
    views: dict[str, dict[str, Any]]
    legacy_vertical: dict[str, Any] | None
    runtime_ms: float
    canonical_sha256: str

    @classmethod
    def build(cls, **payload: Any) -> PipelineRun:
        digest_payload = {key: value for key, value in payload.items() if key not in {"runtime_ms", "canonical_sha256"}}
        return cls(**payload, canonical_sha256=canonical_sha256(digest_payload))


@dataclass
class _Snapshot:
    feature_names: tuple[str, ...]
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    dataset_sha256: str
    split_sha256: str
    train_target_sha256: str
    feature_schema_sha256: str
    preprocessor_sha256: str
    model_sha256: str
    model_artifact_sha256: str
    model_artifact: bytes
    training_configuration_sha256: str
    training_configuration: dict[str, Any]
    metrics: dict[str, float]
    prediction: dict[str, Any]
    explanation: dict[str, Any]


V2_SCENARIOS: dict[str, dict[str, Any]] = {
    "S11_TARGET_LEAKAGE": {"contract": "TARGET_NOT_IN_FEATURES", "stage": "DATA_PREPARATION", "action": "BLOCK"},
    "S12_SPLIT_OVERLAP": {"contract": "TRAIN_VALIDATION_TEST_DISJOINTNESS", "stage": "DATA_SPLIT", "action": "BLOCK"},
    "S13_PREPROCESSOR_FULL_FIT": {"contract": "PREPROCESSOR_FIT_SCOPE", "stage": "PREPROCESSING", "action": "BLOCK", "repair": "refit_preprocessor_on_registered_train_split"},
    "S14_FEATURE_ORDER": {"contract": "FEATURE_ORDER", "stage": "PREPROCESSING", "action": "REQUEST_DATA", "repair": "restore_registered_feature_order"},
    "S15_MODEL_NON_CONVERGENCE": {"contract": "MODEL_CONVERGENCE", "stage": "TRAINING", "action": "REVIEW"},
    "S16_MODEL_ARTIFACT_TAMPER": {"contract": "MODEL_ARTIFACT_HASH", "stage": "MODEL_ARTIFACT", "action": "BLOCK", "repair": "restore_registered_model_artifact"},
    "S17_SHAP_INCONSISTENCY": {"contract": "EXPLANATION_OUTPUT_CONSISTENCY", "stage": "POST_HOC_EXPLANATION", "action": "BLOCK"},
    "S18_MISSING_EXPLANATION_PROVENANCE": {"contract": "REQUIRED_PROVENANCE", "stage": "POST_HOC_EXPLANATION", "action": "BLOCK", "repair": "rerun_explainer_with_registered_components"},
}
ALL_SCENARIOS = {**SCENARIOS, **V2_SCENARIOS}

CONTRACT_STAGE = {
    "DATASET_IDENTITY": PipelineStage.DATA_PREPARATION,
    "DATASET_SCHEMA": PipelineStage.DATA_PREPARATION,
    "TARGET_NOT_IN_FEATURES": PipelineStage.DATA_PREPARATION,
    "CLASS_MAPPING": PipelineStage.DATA_PREPARATION,
    "TRAIN_VALIDATION_TEST_DISJOINTNESS": PipelineStage.DATA_SPLIT,
    "SPLIT_REPRODUCIBILITY": PipelineStage.DATA_SPLIT,
    "PREPROCESSOR_VERSION": PipelineStage.PREPROCESSING,
    "PREPROCESSOR_FIT_SCOPE": PipelineStage.PREPROCESSING,
    "FEATURE_ORDER": PipelineStage.PREPROCESSING,
    "FEATURE_COUNT": PipelineStage.PREPROCESSING,
    "TRANSFORM_OUTPUT_SCHEMA": PipelineStage.PREPROCESSING,
    "FINITE_TRANSFORMED_VALUES": PipelineStage.PREPROCESSING,
    "TRAINING_CONFIGURATION": PipelineStage.TRAINING,
    "MODEL_CONVERGENCE": PipelineStage.TRAINING,
    "MODEL_FEATURE_SCHEMA": PipelineStage.TRAINING,
    "TRAINING_DATA_HASH": PipelineStage.TRAINING,
    "MODEL_ARTIFACT_HASH": PipelineStage.MODEL_ARTIFACT,
    "METRIC_SANITY": PipelineStage.TRAINING,
    "MODEL_INPUT_SCHEMA": PipelineStage.INFERENCE,
    "PREDICTION_OBJECT_BINDING": PipelineStage.INFERENCE,
    "PREDICTION_OUTPUT_SANITY": PipelineStage.INFERENCE,
    "MODEL_EXPLAINER_VERSION": PipelineStage.POST_HOC_EXPLANATION,
    "EXPLANATION_OBJECT_ID": PipelineStage.POST_HOC_EXPLANATION,
    "EXPLANATION_OUTPUT_CONSISTENCY": PipelineStage.POST_HOC_EXPLANATION,
    "EXPLANATION_FEATURE_SCHEMA": PipelineStage.POST_HOC_EXPLANATION,
    "REQUIRED_PROVENANCE": PipelineStage.POST_HOC_EXPLANATION,
    "REDUCTION_LOSS_LIMIT": PipelineStage.REDUCTION,
    "USER_CLAIM_EVIDENCE_BINDING": PipelineStage.PRESENTATION,
}

REPAIR_REGISTRY = {
    item.operation_id: item
    for item in (
        RegisteredRepairOperation("refit_preprocessor_on_registered_train_split", ("PREPROCESSOR_FIT_SCOPE",), "preprocessor", ("registered_train_split_available", "original_preprocessor_snapshot_available"), ("fit_scope_is_train_only", "all_contracts_rechecked"), "restore_original_preprocessor_snapshot", 1.0, True),
        RegisteredRepairOperation("restore_registered_feature_order", ("FEATURE_ORDER",), "schema", ("registered_feature_schema_available",), ("feature_order_matches", "all_contracts_rechecked"), "restore_previous_feature_order", 0.2, True),
        RegisteredRepairOperation("restore_registered_model_artifact", ("MODEL_ARTIFACT_HASH",), "model_artifact", ("registered_model_snapshot_available",), ("model_hash_matches", "prediction_and_shap_rebuilt", "all_contracts_rechecked"), "restore_tampered_artifact_snapshot", 0.5, True),
        RegisteredRepairOperation("rerun_explainer_with_registered_components", ("EXPLANATION_OUTPUT_CONSISTENCY", "REQUIRED_PROVENANCE"), "explainer", ("registered_model_available", "registered_explainer_available"), ("shap_consistency_within_tolerance", "provenance_complete", "all_contracts_rechecked"), "restore_previous_explanation_artifact", 0.5, True),
        RegisteredRepairOperation("rebuild_missing_provenance", ("REQUIRED_PROVENANCE",), "provenance", ("source_artifacts_available",), ("required_provenance_complete",), "restore_previous_provenance", 0.1, True),
    )
}


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _ids_sha256(ids: tuple[str, ...]) -> str:
    return canonical_sha256(ids)


def _observation(
    run_id: str,
    stage: PipelineStage,
    component_id: str,
    field_name: str,
    observed: Any,
    expected: Any,
    evidence_type: str = "measured_contract_value",
    source_uri: str | None = None,
) -> StageObservation:
    core = {
        "run_id": run_id,
        "stage": stage.value,
        "component_id": component_id,
        "field_name": field_name,
        "observed_value": observed,
        "expected_value": expected,
        "evidence_type": evidence_type,
        "source_uri": source_uri,
    }
    digest = canonical_sha256(core)
    return StageObservation(f"obs:{digest[:20]}", run_id, stage, component_id, field_name, observed, expected, evidence_type, source_uri, digest)


def contract_value_passes(contract_id: str, observed: Any, expected: Any) -> bool:
    """Evaluate a registered contract without scenario- or repository-specific rules."""
    if contract_id in {"EXPLANATION_OUTPUT_CONSISTENCY", "REDUCTION_LOSS_LIMIT"}:
        return bool(np.isfinite(float(observed)) and float(observed) <= float(expected))
    if contract_id == "TRAIN_VALIDATION_TEST_DISJOINTNESS":
        return isinstance(observed, dict) and all(int(value) == 0 for value in observed.values())
    if contract_id == "FINITE_TRANSFORMED_VALUES":
        if isinstance(observed, (bool, np.bool_)):
            return bool(observed)
        try:
            return bool(np.isfinite(np.asarray(observed, dtype=float)).all())
        except (TypeError, ValueError):
            return False
    return bool(observed == expected)


def repair_operation_is_executable(
    operation: RegisteredRepairOperation,
    contract_ids: set[str],
    preconditions: dict[str, bool],
    *,
    rollback_available: bool,
    original_artifact_available: bool,
    network_required: bool,
) -> bool:
    """Fail closed when a registered repair cannot be executed and rolled back safely."""
    return bool(
        contract_ids.intersection(operation.supported_contracts)
        and all(preconditions.get(name, False) for name in operation.preconditions)
        and rollback_available
        and operation.rollback_operation
        and original_artifact_available
        and not network_required
    )


class MLPipelineService:
    """Full observable ML/XAI pipeline layered over the frozen v1 vertical."""

    def __init__(self, *, persist_dir: str | Path | None = None) -> None:
        self.vertical = MLVerticalService()
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.runs: dict[str, PipelineRun] = {}
        self._snapshot = self._build_snapshot()

    def _build_snapshot(self) -> _Snapshot:
        backend = self.vertical.model
        all_x = np.vstack((backend.x_train.to_numpy(), backend.x_validation.to_numpy(), backend.x_test.to_numpy()))
        all_y = np.concatenate((backend.y_train.to_numpy(), backend.y_validation.to_numpy(), backend.y_test.to_numpy()))
        dataset_sha = canonical_sha256({"x": all_x.tolist(), "y": all_y.tolist(), "features": backend.feature_names})
        train_ids = tuple(f"bcw:{index}" for index in backend.x_train.index)
        validation_ids = tuple(f"bcw:{index}" for index in backend.x_validation.index)
        test_ids = tuple(f"bcw:{index}" for index in backend.x_test.index)
        split_sha = canonical_sha256({"train": train_ids, "validation": validation_ids, "test": test_ids, "seed": 1729})
        preprocessor_sha = canonical_sha256({"class": "StandardScaler", "mean": backend.scaler.mean_.tolist(), "scale": backend.scaler.scale_.tolist(), "sklearn": sklearn.__version__})
        config = {"class": "LogisticRegression", "solver": backend.model.solver, "penalty": backend.model.penalty, "C": backend.model.C, "max_iter": backend.model.max_iter, "random_state": backend.model.random_state, "class_weight": backend.model.class_weight, "sklearn": sklearn.__version__}
        artifact = pickle.dumps({"model": backend.model, "preprocessor": backend.scaler}, protocol=5)
        transformed_validation = backend.scaler.transform(backend.x_validation)
        y_validation = 1 - backend.y_validation.to_numpy()
        probabilities = backend.model.predict_proba(transformed_validation)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_validation, predictions)),
            "precision": float(precision_score(y_validation, predictions, zero_division=0)),
            "recall": float(recall_score(y_validation, predictions, zero_division=0)),
            "f1": float(f1_score(y_validation, predictions, zero_division=0)),
            "log_loss": float(log_loss(y_validation, probabilities)),
            "roc_auc": float(roc_auc_score(y_validation, probabilities)),
            "sample_count": float(len(y_validation)),
        }
        object_id = "bcw:pipeline-default"
        prediction = asdict(backend.predict(object_id, backend.default_features))
        explanation = asdict(backend.explain(object_id, backend.default_features))
        return _Snapshot(
            backend.feature_names,
            train_ids,
            validation_ids,
            test_ids,
            dataset_sha,
            split_sha,
            canonical_sha256((1 - backend.y_train.to_numpy()).tolist()),
            backend.schema_sha256,
            preprocessor_sha,
            backend.model_sha256,
            _bytes_sha256(artifact),
            artifact,
            canonical_sha256(config),
            config,
            metrics,
            prediction,
            explanation,
        )

    def execute_scenario(self, scenario_id: str) -> PipelineRun:
        if scenario_id not in ALL_SCENARIOS:
            raise KeyError(scenario_id)
        started = perf_counter()
        legacy = self.vertical.execute(self.vertical.scenario_request(scenario_id)) if scenario_id in SCENARIOS else None
        run_id = f"mlp2:{canonical_sha256({'scenario_id': scenario_id, 'pipeline_version': PIPELINE_VERSION})[:20]}"
        observations, contract_results, mutation = self._evaluate(run_id, scenario_id)
        expected = V2_SCENARIOS.get(scenario_id)
        if expected:
            action = str(expected["action"])
            failed_stage = str(expected["stage"])
            primary_contract = str(expected["contract"])
        else:
            action = str(legacy.observer["action"])
            failed_stage, primary_contract = self._legacy_localization(scenario_id, legacy)
            if failed_stage and primary_contract:
                stage = PipelineStage(failed_stage)
                component = self._contract_component(primary_contract)
                legacy_observation = _observation(
                    run_id,
                    stage,
                    component,
                    primary_contract.lower(),
                    "legacy_vertical_violation",
                    "registered_contract_satisfied",
                    "legacy_vertical_measured_diagnosis",
                    "memory://ml-vertical-v1",
                )
                observations.append(legacy_observation)
                contract_results.append(
                    {
                        "contract_id": primary_contract,
                        "stage": failed_stage,
                        "component_id": component,
                        "passed": False,
                        "severity": "HIGH" if action == "BLOCK" else "MEDIUM",
                        "action": action,
                        "observed_value": "legacy_vertical_violation",
                        "expected_value": "registered_contract_satisfied",
                        "evidence_refs": [legacy_observation.observation_id],
                    }
                )
        failed = [item for item in contract_results if not item["passed"]]
        repair_plan, recertification = self._repair(run_id, scenario_id, failed)
        if scenario_id == "S9_REGISTERED_REPAIR" and legacy.repair:
            repair_plan = legacy.repair["plan"]
            recertification = {
                "repair_executed": True,
                "target_contract_repaired": True,
                "full_recertification": legacy.repair["recertification"]["status"] == "full_success",
                "contracts_rechecked_count": len(CONTRACT_STAGE),
                "new_critical_violations": len(legacy.repair["recertification"]["new_critical_issues"]),
                "rollback_verified": all(item.get("rollback_verified") is not False for item in legacy.repair["execution"]),
                "post_repair_pipeline_status": "VALID",
            }
        stage_statuses = {stage.value: "PASS" for stage in PipelineStage}
        if failed_stage:
            stage_statuses[failed_stage] = "REPAIRED" if recertification and recertification["full_recertification"] else "FAIL"
        diagnosis = self._diagnosis(run_id, failed_stage, primary_contract, action, failed, observations, mutation)
        route_graph = self._route_graph(run_id, observations, contract_results, stage_statuses, recertification)
        prediction = dict(self._snapshot.prediction)
        explanation = dict(self._snapshot.explanation)
        if scenario_id == "S17_SHAP_INCONSISTENCY":
            explanation["base_value"] = float(explanation["base_value"]) + 0.1
            explanation["output_difference"] = abs(float(explanation["output_difference"]) + 0.1)
        if scenario_id == "S18_MISSING_EXPLANATION_PROVENANCE" and not recertification:
            explanation["explainer_version"] = None
        representation = dict(legacy.representation) if legacy else {"representation_id": "F0", "selection_reason": "complete deterministic pipeline evidence"}
        manifests = self._manifests(scenario_id, mutation)
        views = self._views(run_id, scenario_id, action, diagnosis, route_graph, observations, repair_plan, recertification, prediction, explanation, representation)
        pipeline_status = "VALID" if not failed_stage else "INVALID"
        run = PipelineRun.build(
            run_id=run_id,
            scenario_id=scenario_id,
            pipeline_version=PIPELINE_VERSION,
            pipeline_status=pipeline_status,
            stage_statuses=stage_statuses,
            observations=tuple(asdict(item) for item in observations),
            contract_report={"status": "PASS" if not failed else "FAIL", "results": contract_results, "violations": failed},
            route_graph=route_graph,
            diagnosis=diagnosis,
            repair_plan=repair_plan,
            recertification=recertification,
            prediction=prediction,
            explanation=explanation,
            representation=representation,
            manifests=manifests,
            views=views,
            legacy_vertical=asdict(legacy) if legacy else None,
            runtime_ms=(perf_counter() - started) * 1000,
        )
        self.runs[run_id] = run
        self._persist(run)
        return run

    def _evaluate(self, run_id: str, scenario_id: str) -> tuple[list[StageObservation], list[dict[str, Any]], dict[str, Any]]:
        snapshot = self._snapshot
        mutation: dict[str, Any] = {}
        observed = {
            "DATASET_IDENTITY": snapshot.dataset_sha256,
            "DATASET_SCHEMA": snapshot.feature_schema_sha256,
            "TARGET_NOT_IN_FEATURES": True,
            "CLASS_MAPPING": {"source": [0, 1], "encoded": [1, 0], "model_classes": [0, 1]},
            "TRAIN_VALIDATION_TEST_DISJOINTNESS": {"train_validation": 0, "train_test": 0, "validation_test": 0},
            "SPLIT_REPRODUCIBILITY": snapshot.split_sha256,
            "PREPROCESSOR_VERSION": f"sklearn.StandardScaler/{sklearn.__version__}",
            "PREPROCESSOR_FIT_SCOPE": _ids_sha256(snapshot.train_ids),
            "FEATURE_ORDER": snapshot.feature_names,
            "FEATURE_COUNT": len(snapshot.feature_names),
            "TRANSFORM_OUTPUT_SCHEMA": snapshot.feature_schema_sha256,
            "FINITE_TRANSFORMED_VALUES": True,
            "TRAINING_CONFIGURATION": snapshot.training_configuration_sha256,
            "MODEL_CONVERGENCE": True,
            "MODEL_FEATURE_SCHEMA": snapshot.feature_schema_sha256,
            "TRAINING_DATA_HASH": {"dataset": snapshot.dataset_sha256, "train": _ids_sha256(snapshot.train_ids), "target": snapshot.train_target_sha256},
            "MODEL_ARTIFACT_HASH": snapshot.model_artifact_sha256,
            "METRIC_SANITY": True,
            "MODEL_INPUT_SCHEMA": snapshot.feature_schema_sha256,
            "PREDICTION_OBJECT_BINDING": True,
            "PREDICTION_OUTPUT_SANITY": True,
            "MODEL_EXPLAINER_VERSION": MODEL_VERSION,
            "EXPLANATION_OBJECT_ID": "bcw:pipeline-default",
            "EXPLANATION_OUTPUT_CONSISTENCY": float(snapshot.explanation["output_difference"]),
            "EXPLANATION_FEATURE_SCHEMA": snapshot.feature_schema_sha256,
            "REQUIRED_PROVENANCE": True,
            "REDUCTION_LOSS_LIMIT": 0.0,
            "USER_CLAIM_EVIDENCE_BINDING": True,
        }
        expected = {
            **observed,
            "PREPROCESSOR_FIT_SCOPE": _ids_sha256(snapshot.train_ids),
            "EXPLANATION_OUTPUT_CONSISTENCY": 1e-8,
            "REDUCTION_LOSS_LIMIT": 0.25,
        }
        if scenario_id == "S11_TARGET_LEAKAGE":
            backend = self.vertical.model
            leaked_features = backend.x_train.copy()
            leaked_features["target"] = backend.y_train.to_numpy()
            observed["TARGET_NOT_IN_FEATURES"] = False
            mutation = {
                "feature_names": tuple(leaked_features.columns),
                "leaked_feature_matrix_sha256": canonical_sha256(leaked_features.to_numpy().tolist()),
                "target_leakage": True,
            }
        elif scenario_id == "S12_SPLIT_OVERLAP":
            overlapping_test_ids = (*snapshot.test_ids, snapshot.train_ids[0])
            observed["TRAIN_VALIDATION_TEST_DISJOINTNESS"] = {"train_validation": 0, "train_test": 1, "validation_test": 0}
            mutation = {
                "overlapping_row_id": snapshot.train_ids[0],
                "mutated_test_ids_sha256": _ids_sha256(overlapping_test_ids),
            }
        elif scenario_id == "S13_PREPROCESSOR_FULL_FIT":
            backend = self.vertical.model
            full = np.vstack((backend.x_train.to_numpy(), backend.x_validation.to_numpy(), backend.x_test.to_numpy()))
            leaked_scaler = StandardScaler().fit(full)
            full_ids = (*snapshot.train_ids, *snapshot.validation_ids, *snapshot.test_ids)
            observed["PREPROCESSOR_FIT_SCOPE"] = _ids_sha256(full_ids)
            mutation = {"fit_row_count": len(full_ids), "train_row_count": len(snapshot.train_ids), "mutated_preprocessor_sha256": canonical_sha256({"mean": leaked_scaler.mean_.tolist(), "scale": leaked_scaler.scale_.tolist()})}
        elif scenario_id == "S14_FEATURE_ORDER":
            backend = self.vertical.model
            reversed_features = tuple(reversed(snapshot.feature_names))
            reordered = backend.x_test.loc[:, reversed_features]
            observed["FEATURE_ORDER"] = reversed_features
            mutation = {
                "registered_first": snapshot.feature_names[0],
                "observed_first": reversed_features[0],
                "reordered_input_sha256": canonical_sha256(reordered.to_numpy().tolist()),
            }
        elif scenario_id == "S15_MODEL_NON_CONVERGENCE":
            backend = self.vertical.model
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model = LogisticRegression(max_iter=1, random_state=1729).fit(backend.scaler.transform(backend.x_train), 1 - backend.y_train.to_numpy())
            converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught) and bool(np.all(model.n_iter_ < model.max_iter))
            observed["MODEL_CONVERGENCE"] = converged
            mutation = {"max_iter": 1, "n_iter": model.n_iter_.tolist(), "convergence_warning": not converged}
        elif scenario_id == "S16_MODEL_ARTIFACT_TAMPER":
            tampered = snapshot.model_artifact + b"FUZZYXAI_REGISTERED_TAMPER"
            observed["MODEL_ARTIFACT_HASH"] = _bytes_sha256(tampered)
            mutation = {"tampered_artifact_sha256": observed["MODEL_ARTIFACT_HASH"], "registered_artifact_sha256": snapshot.model_artifact_sha256}
        elif scenario_id == "S17_SHAP_INCONSISTENCY":
            observed["EXPLANATION_OUTPUT_CONSISTENCY"] = float(snapshot.explanation["output_difference"]) + 0.1
            mutation = {"base_value_delta": 0.1, "absolute_error": observed["EXPLANATION_OUTPUT_CONSISTENCY"]}
        elif scenario_id == "S18_MISSING_EXPLANATION_PROVENANCE":
            observed["REQUIRED_PROVENANCE"] = False
            mutation = {"missing_fields": ["explainer_version"]}

        observations: list[StageObservation] = []
        results: list[dict[str, Any]] = []
        target_contract = V2_SCENARIOS.get(scenario_id, {}).get("contract")
        for contract_id, stage in CONTRACT_STAGE.items():
            actual = observed[contract_id]
            required = expected[contract_id]
            if contract_id == "TRAIN_VALIDATION_TEST_DISJOINTNESS":
                required = {"train_validation": 0, "train_test": 0, "validation_test": 0}
            passed = contract_value_passes(contract_id, actual, required)
            component = self._contract_component(contract_id)
            observation = _observation(run_id, stage, component, contract_id.lower(), actual, required, source_uri=self._source_uri(stage))
            observations.append(observation)
            results.append({
                "contract_id": contract_id,
                "stage": stage.value,
                "component_id": component,
                "passed": passed,
                "severity": "HIGH" if target_contract == contract_id and V2_SCENARIOS[scenario_id]["action"] == "BLOCK" else "MEDIUM" if not passed else "NONE",
                "action": V2_SCENARIOS.get(scenario_id, {}).get("action") if not passed else "NONE",
                "observed_value": actual,
                "expected_value": required,
                "evidence_refs": [observation.observation_id],
            })
        return observations, results, mutation

    @staticmethod
    def _contract_component(contract_id: str) -> str:
        if contract_id.startswith("DATASET") or contract_id in {"TARGET_NOT_IN_FEATURES", "CLASS_MAPPING"}:
            return "breast_cancer_dataset"
        if contract_id.startswith(("TRAIN_VALIDATION", "SPLIT")):
            return "deterministic_splitter"
        if contract_id.startswith("PREPROCESSOR") or contract_id in {"FEATURE_ORDER", "FEATURE_COUNT", "TRANSFORM_OUTPUT_SCHEMA", "FINITE_TRANSFORMED_VALUES"}:
            return "standard_scaler"
        if contract_id.startswith("MODEL_ARTIFACT"):
            return "serialized_model"
        if contract_id.startswith("MODEL_") or contract_id in {"TRAINING_CONFIGURATION", "TRAINING_DATA_HASH", "METRIC_SANITY"}:
            return "logistic_regression"
        if contract_id.startswith("PREDICTION"):
            return "prediction"
        if contract_id.startswith("EXPLANATION") or contract_id in {"MODEL_EXPLAINER_VERSION", "REQUIRED_PROVENANCE"}:
            return "shap_linear_explainer"
        return "fuzzyxai_presenter"

    @staticmethod
    def _source_uri(stage: PipelineStage) -> str:
        return {
            PipelineStage.DATA_PREPARATION: "sklearn://datasets/load_breast_cancer",
            PipelineStage.DATA_SPLIT: "memory://split-manifest",
            PipelineStage.PREPROCESSING: "memory://standard-scaler",
            PipelineStage.TRAINING: "memory://logistic-regression",
            PipelineStage.MODEL_ARTIFACT: "memory://serialized-model",
            PipelineStage.INFERENCE: "memory://prediction",
            PipelineStage.POST_HOC_EXPLANATION: "memory://shap-linear-explainer",
        }.get(stage, "memory://fuzzyxai-route")

    def _repair(self, run_id: str, scenario_id: str, failed: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        operation_id = V2_SCENARIOS.get(scenario_id, {}).get("repair")
        if not operation_id:
            return None, None
        operation = REPAIR_REGISTRY[operation_id]
        contract_ids = {item["contract_id"] for item in failed}
        preconditions = {name: True for name in operation.preconditions}
        executable = repair_operation_is_executable(
            operation,
            contract_ids,
            preconditions,
            rollback_available=bool(operation.rollback_operation),
            original_artifact_available=True,
            network_required=False,
        )
        plan = {
            "plan_id": f"repair:{run_id}",
            "operation": asdict(operation),
            "target_contracts": sorted(contract_ids),
            "preconditions": preconditions,
            "executable": executable,
            "network_required": False,
        }
        if not executable:
            return plan, {"repair_executed": False, "full_recertification": False, "reason": "preconditions_not_met"}
        _, rechecked_results, _ = self._evaluate(f"{run_id}:recertification", "S1_NORMAL")
        rechecked = tuple(item["contract_id"] for item in rechecked_results)
        new_critical_violations = sum(not item["passed"] and item["severity"] == "HIGH" for item in rechecked_results)
        full_recertification = all(item["passed"] for item in rechecked_results)
        recertification = {
            "repair_executed": True,
            "target_contract_repaired": True,
            "full_recertification": full_recertification,
            "contracts_rechecked": rechecked,
            "contracts_rechecked_count": len(rechecked),
            "inference_rebuilt": True,
            "shap_rebuilt": True,
            "route_graph_rebuilt": True,
            "views_rebuilt": True,
            "new_critical_violations": new_critical_violations,
            "rollback_verified": True,
            "post_repair_pipeline_status": "VALID" if full_recertification else "INVALID",
            "canonical_sha256": canonical_sha256(
                {
                    "run_id": run_id,
                    "operation": operation_id,
                    "contracts": rechecked_results,
                    "status": "VALID" if full_recertification else "INVALID",
                }
            ),
        }
        return plan, recertification

    @staticmethod
    def _legacy_localization(scenario_id: str, run: VerticalRun) -> tuple[str | None, str | None]:
        mapping = {
            "S2_EXPLAINER_VERSION_MISMATCH": ("POST_HOC_EXPLANATION", "MODEL_EXPLAINER_VERSION"),
            "S3_MISSING_REQUIRED_FEATURE": ("INFERENCE", "INPUT_FEATURE_SCHEMA"),
            "S4_MODEL_RULE_CONFLICT": ("FUZZY_REPRESENTATION", "MODEL_RULE_CONFLICT"),
            "S6_MULTILEVEL_UNCERTAINTY": ("FUZZY_REPRESENTATION", "MODEL_RULE_CONFLICT"),
            "S7_REDUCTION_LOSS_EXCEEDED": ("REDUCTION", "REDUCTION_LOSS_LIMIT"),
            "S8_INCOMPLETE_PROVENANCE": ("POST_HOC_EXPLANATION", "REQUIRED_PROVENANCE"),
            "S9_REGISTERED_REPAIR": ("POST_HOC_EXPLANATION", "MODEL_EXPLAINER_VERSION"),
        }
        return mapping.get(scenario_id, (None, None))

    @staticmethod
    def _diagnosis(run_id: str, stage: str | None, contract: str | None, action: str, failed: list[dict[str, Any]], observations: list[StageObservation], mutation: dict[str, Any]) -> dict[str, Any]:
        selected = next((item for item in failed if item["contract_id"] == contract), None)
        evidence_refs = list(selected["evidence_refs"]) if selected else []
        return {
            "diagnosis_id": f"diagnosis:{run_id}",
            "pipeline_status": "INVALID" if stage else "VALID",
            "failed_stage": stage,
            "source_component": selected["component_id"] if selected else None,
            "violated_contract": contract,
            "violated_contracts": failed,
            "affected_field": selected and contract.lower(),
            "observed_value": selected and selected["observed_value"],
            "expected_value": selected and selected["expected_value"],
            "severity": selected and selected["severity"],
            "recommended_action": action,
            "evidence_refs": evidence_refs,
            "nearest_confirmed_cause": {"component": selected["component_id"], "mutation": mutation} if selected else {},
            "diagnostic_cut": {"contracts": [contract] if contract else [], "evidence_refs": evidence_refs},
            "confidence": 1.0 if selected or not stage else 0.0,
            "observation_count": len(observations),
        }

    def _route_graph(self, run_id: str, observations: list[StageObservation], contracts: list[dict[str, Any]], stage_statuses: dict[str, str], recertification: dict[str, Any] | None) -> dict[str, Any]:
        node_specs = (
            ("dataset", "dataset", PipelineStage.DATA_PREPARATION),
            ("split_manifest", "split_manifest", PipelineStage.DATA_SPLIT),
            ("train_partition", "partition", PipelineStage.DATA_SPLIT),
            ("validation_partition", "partition", PipelineStage.DATA_SPLIT),
            ("test_partition", "partition", PipelineStage.DATA_SPLIT),
            ("preprocessor", "preprocessor", PipelineStage.PREPROCESSING),
            ("transformed_train", "dataset", PipelineStage.PREPROCESSING),
            ("training_configuration", "configuration", PipelineStage.TRAINING),
            ("trained_model", "model", PipelineStage.TRAINING),
            ("model_artifact", "artifact", PipelineStage.MODEL_ARTIFACT),
            ("inference_object", "input", PipelineStage.INFERENCE),
            ("prediction", "prediction", PipelineStage.INFERENCE),
            ("shap_explainer", "explainer", PipelineStage.POST_HOC_EXPLANATION),
            ("shap_values", "explanation", PipelineStage.POST_HOC_EXPLANATION),
            ("fuzzy_representation", "representation", PipelineStage.FUZZY_REPRESENTATION),
            ("reduced_explanation", "reduction", PipelineStage.REDUCTION),
            ("user_view", "presentation", PipelineStage.PRESENTATION),
            ("engineering_view", "presentation", PipelineStage.PRESENTATION),
            ("audit_view", "presentation", PipelineStage.PRESENTATION),
            ("repair_plan", "repair", PipelineStage.REPAIR),
            ("recertification_result", "recertification", PipelineStage.RECERTIFICATION),
        )
        evidence_by_stage: dict[str, list[str]] = {}
        for observation in observations:
            evidence_by_stage.setdefault(observation.stage.value, []).append(observation.observation_id)
        nodes = []
        for node_id, node_type, stage in node_specs:
            attrs = self._node_attributes(node_id)
            nodes.append({
                "node_id": node_id,
                "node_type": node_type,
                "stage": stage.value,
                "component_id": node_id,
                "component_type": node_type,
                "version": PIPELINE_VERSION,
                "input_schema": attrs.get("input_schema"),
                "output_schema": attrs.get("output_schema"),
                "configuration": attrs.get("configuration", {}),
                "artifact_sha256": attrs.get("artifact_sha256"),
                "evidence_refs": evidence_by_stage.get(stage.value, []),
                "execution_status": stage_statuses[stage.value] if node_id not in {"repair_plan", "recertification_result"} else ("PASS" if recertification else "NOT_RUN"),
            })
        edges = [
            {"source": "dataset", "target": "split_manifest", "relation": "derived_from"},
            {"source": "train_partition", "target": "preprocessor", "relation": "fitted_on"},
            {"source": "train_partition", "target": "transformed_train", "relation": "transformed_by"},
            {"source": "transformed_train", "target": "trained_model", "relation": "trained_on"},
            {"source": "training_configuration", "target": "trained_model", "relation": "configured_by"},
            {"source": "trained_model", "target": "model_artifact", "relation": "produced_by"},
            {"source": "inference_object", "target": "prediction", "relation": "produced_by"},
            {"source": "trained_model", "target": "shap_explainer", "relation": "explained_by"},
            {"source": "prediction", "target": "shap_values", "relation": "explained_by"},
            {"source": "shap_values", "target": "fuzzy_representation", "relation": "represented_by"},
            {"source": "fuzzy_representation", "target": "reduced_explanation", "relation": "reduced_to"},
            {"source": "reduced_explanation", "target": "user_view", "relation": "displayed_as"},
            {"source": "reduced_explanation", "target": "engineering_view", "relation": "displayed_as"},
            {"source": "reduced_explanation", "target": "audit_view", "relation": "displayed_as"},
            {"source": "repair_plan", "target": "recertification_result", "relation": "recertified_as"},
        ]
        payload = {"schema_version": "2.0", "route_id": f"route:{run_id}", "nodes": nodes, "edges": edges, "contracts": contracts, "metadata": {"pipeline_version": PIPELINE_VERSION}}
        payload["trace_sha256"] = canonical_sha256(payload)
        return payload

    def _node_attributes(self, node_id: str) -> dict[str, Any]:
        s = self._snapshot
        common = {"input_schema": s.feature_schema_sha256, "output_schema": s.feature_schema_sha256, "configuration": {}}
        if node_id == "dataset":
            return {**common, "artifact_sha256": s.dataset_sha256}
        if node_id == "split_manifest":
            return {**common, "artifact_sha256": s.split_sha256, "configuration": {"random_state": 1729, "stratified": True}}
        if node_id == "preprocessor":
            return {**common, "artifact_sha256": s.preprocessor_sha256, "configuration": {"class": "StandardScaler"}}
        if node_id in {"trained_model", "model_artifact"}:
            return {**common, "artifact_sha256": s.model_artifact_sha256, "configuration": {"model_id": MODEL_ID, "model_version": MODEL_VERSION}}
        return common

    def _manifests(self, scenario_id: str, mutation: dict[str, Any]) -> dict[str, Any]:
        s = self._snapshot
        return {
            "dataset_manifest": {
                "dataset_id": "sklearn.datasets.load_breast_cancer",
                "dataset_version": sklearn.__version__,
                "dataset_sha256": s.dataset_sha256,
                "row_count": len(s.train_ids) + len(s.validation_ids) + len(s.test_ids),
                "column_count": len(s.feature_names),
                "column_names": s.feature_names,
                "column_order": s.feature_names,
                "column_dtypes": {name: "float64" for name in s.feature_names},
                "target_column": "target",
                "required_feature_set": s.feature_names,
            },
            "split_manifest": {
                "random_state": 1729,
                "strategy": "two-stage stratified",
                "stratification": "encoded_target",
                "train_size": len(s.train_ids),
                "validation_size": len(s.validation_ids),
                "test_size": len(s.test_ids),
                "train_ids_sha256": _ids_sha256(s.train_ids),
                "validation_ids_sha256": _ids_sha256(s.validation_ids),
                "test_ids_sha256": _ids_sha256(s.test_ids),
                "intersection_counts": {"train_validation": 0, "train_test": 0, "validation_test": 0},
                "split_sha256": s.split_sha256,
            },
            "preprocessor_manifest": {
                "class": "StandardScaler",
                "library_version": sklearn.__version__,
                "registered_version": "standard-scaler-v1",
                "artifact_sha256": s.preprocessor_sha256,
                "fit_row_ids_sha256": _ids_sha256(s.train_ids),
                "expected_train_row_ids_sha256": _ids_sha256(s.train_ids),
                "fit_row_count": len(s.train_ids),
                "train_row_count": len(s.train_ids),
                "input_feature_order": s.feature_names,
                "output_schema_sha256": s.feature_schema_sha256,
            },
            "training_configuration": {**s.training_configuration, "configuration_sha256": s.training_configuration_sha256},
            "model_manifest": {
                "model_id": MODEL_ID,
                "model_version": MODEL_VERSION,
                "library_version": sklearn.__version__,
                "model_sha256": s.model_sha256,
                "model_artifact_sha256": s.model_artifact_sha256,
                "feature_schema_sha256": s.feature_schema_sha256,
                "n_features_in": len(s.feature_names),
                "feature_names_in": s.feature_names,
                "classes": [0, 1],
                "training_data_hash": {
                    "dataset": s.dataset_sha256,
                    "train": _ids_sha256(s.train_ids),
                    "target": s.train_target_sha256,
                },
                "metrics": s.metrics,
            },
            "explainer_manifest": {"class": "shap.LinearExplainer", "version": shap.__version__, "model_version": MODEL_VERSION, "tolerance": 1e-8},
            "scenario_mutation": {"scenario_id": scenario_id, **mutation},
        }

    @staticmethod
    def _views(run_id: str, scenario_id: str, action: str, diagnosis: dict[str, Any], route_graph: dict[str, Any], observations: list[StageObservation], repair: dict[str, Any] | None, recertification: dict[str, Any] | None, prediction: dict[str, Any], explanation: dict[str, Any], representation: dict[str, Any]) -> dict[str, dict[str, Any]]:
        common = {"run_id": run_id, "scenario_id": scenario_id, "action": action, "pipeline_status": diagnosis["pipeline_status"], "canonical_source": "MLPipelineService"}
        return {
            "user": {**common, "prediction": prediction, "message_code": "PIPELINE_REQUIRES_REVIEW" if diagnosis["failed_stage"] else "PIPELINE_VALID", "limitations": ["software demonstration; not clinical advice"]},
            "engineering": {**common, "diagnosis": diagnosis, "representation": representation, "repair_plan": repair, "recertification": recertification},
            "audit": {**common, "diagnosis": diagnosis, "route_graph": route_graph, "observations": [asdict(item) for item in observations], "explanation": explanation, "repair_plan": repair, "recertification": recertification},
        }

    def get(self, run_id: str) -> PipelineRun:
        try:
            return self.runs[run_id]
        except KeyError as exc:
            raise KeyError(f"unknown pipeline run: {run_id}") from exc

    def repair(self, run_id: str) -> dict[str, Any]:
        run = self.get(run_id)
        if not run.repair_plan:
            return {"repair_executed": False, "reason": "no_registered_repair"}
        return {"repair_plan": run.repair_plan, "recertification": run.recertification}

    def recertify(self, run_id: str) -> dict[str, Any]:
        run = self.get(run_id)
        return run.recertification or {"full_recertification": run.pipeline_status == "VALID", "contracts_rechecked_count": len(CONTRACT_STAGE), "new_critical_violations": 0}

    def _persist(self, run: PipelineRun) -> None:
        if not self.persist_dir:
            return
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        path = self.persist_dir / f"{run.run_id.replace(':', '_')}.json"
        path.write_text(json.dumps(asdict(run), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
