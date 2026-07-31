from __future__ import annotations

import inspect
from dataclasses import asdict
from pathlib import Path

from fuzzyxai.ml_vertical.comparative import (
    CROSS_STAGE_CONTRACTS,
    LOCAL_CONTRACTS,
    MODE_IDS,
    ModeInput,
    evaluate_mode,
    project_mode_input,
)
from fuzzyxai.ml_vertical.pipeline import MLPipelineService


def test_b0_receives_only_standard_log(pipeline_service: MLPipelineService) -> None:
    mode_input = project_mode_input(pipeline_service.execute_scenario("S13_PREPROCESSOR_FULL_FIT"), "B0")

    assert mode_input.standard_log is not None
    assert mode_input.route_graph is None
    assert not mode_input.contract_results
    assert mode_input.full_run is None


def test_b1_reads_only_registered_mlflow_observation(pipeline_service: MLPipelineService) -> None:
    mode_input = project_mode_input(pipeline_service.execute_scenario("S17_SHAP_INCONSISTENCY"), "B1")
    payload = asdict(mode_input.mlflow_observation)

    assert mode_input.mlflow_observation is not None
    assert set(payload) == {"parameters", "metrics", "tags", "registered_artifacts"}
    assert not {"route_graph", "diagnosis", "contract_report", "pipeline_valid"}.intersection(str(payload))
    assert mode_input.full_run is None


def test_b2_excludes_route_graph_and_cross_stage_contracts(pipeline_service: MLPipelineService) -> None:
    mode_input = project_mode_input(pipeline_service.execute_scenario("S13_PREPROCESSOR_FULL_FIT"), "B2")

    assert mode_input.route_graph is None
    assert all(item["contract_id"] in LOCAL_CONTRACTS for item in mode_input.contract_results)
    assert not CROSS_STAGE_CONTRACTS.intersection(item["contract_id"] for item in mode_input.contract_results)
    assert not evaluate_mode(mode_input).detected


def test_b3_uses_fixed_greedy_order() -> None:
    observations = (
        _observation("obs:z", "TRAINING", "z", "z"),
        _observation("obs:a", "DATA_SPLIT", "a", "a"),
    )
    contracts = (
        _contract("Z_CONTRACT", "TRAINING", "z", "MEDIUM", "obs:z"),
        _contract("A_CONTRACT", "DATA_SPLIT", "a", "HIGH", "obs:a"),
    )

    result = evaluate_mode(ModeInput("fixture", "B3", observations=observations, contract_results=contracts))

    assert result.contract_id == "A_CONTRACT"
    assert result.stage == "DATA_SPLIT"


def test_a1_has_graph_but_not_cross_stage_checks(pipeline_service: MLPipelineService) -> None:
    mode_input = project_mode_input(pipeline_service.execute_scenario("S12_SPLIT_OVERLAP"), "A1")

    assert mode_input.route_graph is not None
    assert not any(item["contract_id"] == "TRAIN_VALIDATION_TEST_DISJOINTNESS" for item in mode_input.contract_results)
    assert not evaluate_mode(mode_input).detected


def test_a2_adds_cross_stage_checks_without_diagnostic_cut(pipeline_service: MLPipelineService) -> None:
    mode_input = project_mode_input(pipeline_service.execute_scenario("S12_SPLIT_OVERLAP"), "A2")
    result = evaluate_mode(mode_input)

    assert result.detected
    assert result.contract_id == "TRAIN_VALIDATION_TEST_DISJOINTNESS"
    assert result.diagnostic_cut is None


def test_a3_builds_cut_and_does_not_execute_repair(pipeline_service: MLPipelineService) -> None:
    result = evaluate_mode(project_mode_input(pipeline_service.execute_scenario("S13_PREPROCESSOR_FULL_FIT"), "A3"))

    assert result.diagnostic_cut
    assert result.repair_available
    assert not result.repair_executed
    assert not result.recertified


def test_a4_executes_full_recertification(pipeline_service: MLPipelineService) -> None:
    result = evaluate_mode(project_mode_input(pipeline_service.execute_scenario("S13_PREPROCESSOR_FULL_FIT"), "A4"))

    assert result.repair_executed
    assert result.target_contract_repaired
    assert result.recertified
    assert result.contracts_rechecked_count == 28
    assert result.new_critical_violations == 0
    assert result.rollback_verified


def test_all_modes_project_the_same_canonical_scenario_run(pipeline_service: MLPipelineService) -> None:
    run = pipeline_service.execute_scenario("S16_MODEL_ARTIFACT_TAMPER")
    projections = [project_mode_input(run, mode_id) for mode_id in MODE_IDS]

    assert {item.scenario_id for item in projections} == {run.scenario_id}
    assert projections[-1].full_run is run
    assert all(item.full_run is None for item in projections[:-1])


def test_gold_fields_cannot_enter_mode_input_or_evaluator() -> None:
    forbidden = {"gold", "target", "expected_answer", "expected_stage", "expected_contract"}

    assert forbidden.isdisjoint(ModeInput.__dataclass_fields__)
    assert tuple(inspect.signature(evaluate_mode).parameters) == ("mode_input",)


def test_disabling_cross_stage_contracts_changes_result(pipeline_service: MLPipelineService) -> None:
    run = pipeline_service.execute_scenario("S13_PREPROCESSOR_FULL_FIT")

    assert not evaluate_mode(project_mode_input(run, "A1")).detected
    assert evaluate_mode(project_mode_input(run, "A2")).contract_id == "PREPROCESSOR_FIT_SCOPE"


def test_mode_runtime_excludes_pipeline_execution(pipeline_service: MLPipelineService) -> None:
    run = pipeline_service.execute_scenario("S1_NORMAL")
    result = evaluate_mode(project_mode_input(run, "A4"))

    assert result.runtime_ms >= 0.0
    assert result.runtime_ms < run.runtime_ms


def test_mode_implementation_has_no_scenario_specific_rules() -> None:
    source = Path(inspect.getsourcefile(evaluate_mode)).read_text(encoding="utf-8")

    assert "S11_" not in source
    assert "S18_" not in source
    assert "scenario_id ==" not in source


def test_result_status_is_derived_and_canonical() -> None:
    result = evaluate_mode(ModeInput("fixture", "B1"))
    payload = asdict(result)

    assert result.pipeline_status == "INSUFFICIENT_EVIDENCE"
    assert result.canonical_sha256
    assert "supported" not in payload
    assert "final_status" not in payload


def test_comparative_layer_does_not_modify_parent_pipeline_api() -> None:
    public = set(inspect.signature(MLPipelineService.execute_scenario).parameters)

    assert public == {"self", "scenario_id"}
    assert not hasattr(MLPipelineService, "comparative_mode")


def _observation(ref: str, stage: str, component: str, value: str) -> dict[str, object]:
    return {
        "observation_id": ref,
        "stage": stage,
        "component_id": component,
        "observed_value": value,
        "expected_value": "expected",
        "source_uri": "memory://fixture",
        "sha256": ref.removeprefix("obs:"),
    }


def _contract(contract_id: str, stage: str, component: str, severity: str, ref: str) -> dict[str, object]:
    return {
        "contract_id": contract_id,
        "stage": stage,
        "component_id": component,
        "passed": False,
        "severity": severity,
        "action": "BLOCK",
        "observed_value": "observed",
        "expected_value": "expected",
        "evidence_refs": [ref],
    }
