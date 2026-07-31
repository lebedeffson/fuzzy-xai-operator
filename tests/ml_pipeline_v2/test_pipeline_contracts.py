from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fuzzyxai.ml_vertical.pipeline import (
    ALL_SCENARIOS,
    CONTRACT_STAGE,
    REPAIR_REGISTRY,
    MLPipelineService,
    contract_value_passes,
    repair_operation_is_executable,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("scenario_id", "stage", "contract_id", "action"),
    (
        ("S11_TARGET_LEAKAGE", "DATA_PREPARATION", "TARGET_NOT_IN_FEATURES", "BLOCK"),
        ("S12_SPLIT_OVERLAP", "DATA_SPLIT", "TRAIN_VALIDATION_TEST_DISJOINTNESS", "BLOCK"),
        ("S13_PREPROCESSOR_FULL_FIT", "PREPROCESSING", "PREPROCESSOR_FIT_SCOPE", "BLOCK"),
        ("S14_FEATURE_ORDER", "PREPROCESSING", "FEATURE_ORDER", "REQUEST_DATA"),
        ("S15_MODEL_NON_CONVERGENCE", "TRAINING", "MODEL_CONVERGENCE", "REVIEW"),
        ("S16_MODEL_ARTIFACT_TAMPER", "MODEL_ARTIFACT", "MODEL_ARTIFACT_HASH", "BLOCK"),
        ("S17_SHAP_INCONSISTENCY", "POST_HOC_EXPLANATION", "EXPLANATION_OUTPUT_CONSISTENCY", "BLOCK"),
        ("S18_MISSING_EXPLANATION_PROVENANCE", "POST_HOC_EXPLANATION", "REQUIRED_PROVENANCE", "BLOCK"),
    ),
)
def test_registered_v2_scenario_localizes_stage_contract_and_action(
    pipeline_service: MLPipelineService,
    scenario_id: str,
    stage: str,
    contract_id: str,
    action: str,
) -> None:
    run = pipeline_service.execute_scenario(scenario_id)

    assert run.pipeline_status == "INVALID"
    assert run.diagnosis["failed_stage"] == stage
    assert run.diagnosis["violated_contract"] == contract_id
    assert run.diagnosis["recommended_action"] == action
    assert run.diagnosis["evidence_refs"]
    assert any(item["contract_id"] == contract_id and not item["passed"] for item in run.contract_report["results"])


def test_feature_count_and_finite_transform_contracts_fail_closed() -> None:
    assert contract_value_passes("FEATURE_COUNT", 30, 30)
    assert not contract_value_passes("FEATURE_COUNT", 29, 30)
    assert contract_value_passes("FINITE_TRANSFORMED_VALUES", np.array([0.0, 1.0]), True)
    assert not contract_value_passes("FINITE_TRANSFORMED_VALUES", np.array([0.0, np.nan]), True)
    assert not contract_value_passes("FINITE_TRANSFORMED_VALUES", np.array([np.inf]), True)


def test_training_data_hash_and_real_model_manifests_are_bound(pipeline_service: MLPipelineService) -> None:
    run = pipeline_service.execute_scenario("S1_NORMAL")
    model = run.manifests["model_manifest"]

    assert model["training_data_hash"]["dataset"] == run.manifests["dataset_manifest"]["dataset_sha256"]
    assert model["training_data_hash"]["train"] == run.manifests["split_manifest"]["train_ids_sha256"]
    assert model["model_artifact_sha256"]
    assert run.manifests["preprocessor_manifest"]["class"] == "StandardScaler"
    assert run.manifests["training_configuration"]["class"] == "LogisticRegression"
    assert run.manifests["explainer_manifest"]["class"] == "shap.LinearExplainer"


def test_repair_requires_registered_contract_preconditions_rollback_and_snapshot() -> None:
    operation = REPAIR_REGISTRY["restore_registered_model_artifact"]
    valid = {name: True for name in operation.preconditions}

    assert repair_operation_is_executable(
        operation,
        {"MODEL_ARTIFACT_HASH"},
        valid,
        rollback_available=True,
        original_artifact_available=True,
        network_required=False,
    )
    assert not repair_operation_is_executable(
        operation,
        {"MODEL_ARTIFACT_HASH"},
        {name: False for name in operation.preconditions},
        rollback_available=True,
        original_artifact_available=True,
        network_required=False,
    )
    assert not repair_operation_is_executable(
        operation,
        {"MODEL_ARTIFACT_HASH"},
        valid,
        rollback_available=False,
        original_artifact_available=True,
        network_required=False,
    )
    assert not repair_operation_is_executable(
        operation,
        {"MODEL_ARTIFACT_HASH"},
        valid,
        rollback_available=True,
        original_artifact_available=False,
        network_required=False,
    )
    assert not repair_operation_is_executable(
        operation,
        {"MODEL_ARTIFACT_HASH"},
        valid,
        rollback_available=True,
        original_artifact_available=True,
        network_required=True,
    )


@pytest.mark.parametrize(
    "scenario_id",
    ("S13_PREPROCESSOR_FULL_FIT", "S14_FEATURE_ORDER", "S16_MODEL_ARTIFACT_TAMPER", "S18_MISSING_EXPLANATION_PROVENANCE"),
)
def test_registered_repairs_recheck_complete_pipeline(pipeline_service: MLPipelineService, scenario_id: str) -> None:
    run = pipeline_service.execute_scenario(scenario_id)
    recertification = run.recertification

    assert run.repair_plan and run.repair_plan["executable"]
    assert recertification and recertification["repair_executed"]
    assert recertification["target_contract_repaired"]
    assert recertification["full_recertification"]
    assert recertification["contracts_rechecked_count"] == len(CONTRACT_STAGE)
    assert set(recertification["contracts_rechecked"]) == set(CONTRACT_STAGE)
    assert recertification["new_critical_violations"] == 0
    assert recertification["rollback_verified"]
    assert recertification["inference_rebuilt"] and recertification["shap_rebuilt"]
    assert recertification["route_graph_rebuilt"] and recertification["views_rebuilt"]


def test_canonical_pipeline_output_is_deterministic(pipeline_service: MLPipelineService) -> None:
    for scenario_id in ALL_SCENARIOS:
        first = pipeline_service.execute_scenario(scenario_id)
        second = pipeline_service.execute_scenario(scenario_id)
        assert first.canonical_sha256 == second.canonical_sha256


def test_v1_scenarios_remain_semantically_identical(pipeline_service: MLPipelineService) -> None:
    for path in sorted((ROOT / "examples/ml_vertical_v1/responses").glob("S*.json")):
        baseline = json.loads(path.read_text(encoding="utf-8"))
        run = pipeline_service.execute_scenario(path.stem)
        assert run.legacy_vertical is not None
        assert _legacy_signature(run.legacy_vertical) == _legacy_signature(baseline)


def _legacy_signature(payload: dict[str, object]) -> tuple[object, ...]:
    diagnosis = payload["diagnosis"]
    return (
        payload["scenario_id"],
        payload["observer"]["action"],
        payload["representation"]["representation_id"],
        diagnosis["route_status"],
        tuple(sorted(item["violated_contract"] for item in diagnosis["issues"])),
    )
