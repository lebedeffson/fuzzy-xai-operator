from __future__ import annotations

from fuzzyxai.ml_vertical.pipeline import CONTRACT_STAGE

from experiments.external_ml_pipeline_v1.benchmark import FAULTS, MODE_IDS, evaluate

PIPELINE = "ext1-sklearn-column-transformer"


def fault(case_id: str):
    return next(item for item in FAULTS if item.case_id == case_id)


def test_correct_route_and_consistent_retrain_pass(benchmark) -> None:
    for case_id, variant in (("C1_BASELINE", "baseline"), ("C2_CONSISTENT_RETRAIN", "retrained")):
        result = evaluate(benchmark.artifacts(PIPELINE, variant), fault(case_id), "O_FUZZYXAI")
        assert not result.detected and result.pipeline_status == "VALID"


def test_each_registered_fault_is_reproduced(benchmark) -> None:
    artifacts = benchmark.artifacts(PIPELINE)
    for item in FAULTS[2:]:
        result = evaluate(artifacts, item, "O_FUZZYXAI")
        assert result.detected and result.contract_id == item.contract_id


def test_local_mode_does_not_use_cross_stage_graph(benchmark) -> None:
    result = evaluate(benchmark.artifacts(PIPELINE), fault("E3_PREPROCESSOR_FIT_SCOPE"), "B_LOCAL_STRONG")
    assert not result.detected and result.diagnostic_cut is None


def test_pairwise_mode_has_no_diagnostic_cut(benchmark) -> None:
    result = evaluate(benchmark.artifacts(PIPELINE), fault("E4_FEATURE_SCHEMA_OR_ORDER"), "B_PAIRWISE_RULES")
    assert result.detected and result.root_cause is None and result.diagnostic_cut is None
    assert result.reported_symptom_count == 5


def test_mlflow_mode_only_sees_registered_tracking_fields(benchmark) -> None:
    artifacts = benchmark.artifacts(PIPELINE)
    assert not evaluate(artifacts, fault("E2_TRAIN_TEST_OVERLAP"), "B_MLFLOW_QUERY").detected
    assert evaluate(artifacts, fault("E5_MODEL_ARTIFACT_MISMATCH"), "B_MLFLOW_QUERY").detected


def test_greedy_mode_selects_first_violation_without_cut(benchmark) -> None:
    result = evaluate(benchmark.artifacts(PIPELINE), fault("E4_FEATURE_SCHEMA_OR_ORDER"), "B_GREEDY_CROSS_STAGE")
    assert result.root_cause == result.contract_id == "FEATURE_ORDER"
    assert result.diagnostic_cut is None and not result.repair_executed


def test_full_mode_selects_one_cascade_source(benchmark) -> None:
    result = evaluate(benchmark.artifacts(PIPELINE), fault("E4_FEATURE_SCHEMA_OR_ORDER"), "O_FUZZYXAI")
    assert result.root_cause == "FEATURE_ORDER"
    assert result.diagnostic_cut and result.diagnostic_cut["size"] == 1
    assert len(result.dependent_violations) == 4


def test_root_repair_removes_dependent_symptoms(benchmark) -> None:
    result = evaluate(benchmark.artifacts(PIPELINE), fault("E4_FEATURE_SCHEMA_OR_ORDER"), "O_FUZZYXAI")
    assert result.repair_executed and result.target_contract_repaired and result.recertified
    assert result.proposed_repair_count == 1 and result.new_critical_violations == 0


def test_recertification_checks_all_existing_contracts(benchmark) -> None:
    result = evaluate(benchmark.artifacts(PIPELINE), fault("E5_MODEL_ARTIFACT_MISMATCH"), "O_FUZZYXAI")
    assert result.contracts_checked == len(CONTRACT_STAGE) == 28


def test_repair_has_registered_rollback(benchmark) -> None:
    result = evaluate(benchmark.artifacts(PIPELINE), fault("E8_EXPLANATION_PROVENANCE"), "O_FUZZYXAI")
    assert result.rollback_verified and result.repair_plan
    assert result.repair_plan["rollback"] == ["restore_previous_artifact_snapshot"]


def test_pairwise_reports_redundant_cascade_repairs(benchmark) -> None:
    result = evaluate(benchmark.artifacts(PIPELINE), fault("E4_FEATURE_SCHEMA_OR_ORDER"), "B_PAIRWISE_RULES")
    assert result.proposed_repair_count == 5 and result.redundant_repair_count == 4


def test_same_mutation_is_seen_by_all_modes(benchmark) -> None:
    artifacts = benchmark.artifacts(PIPELINE)
    results = [evaluate(artifacts, fault("E5_MODEL_ARTIFACT_MISMATCH"), mode) for mode in MODE_IDS]
    assert {item.contract_id for item in results} == {"MODEL_ARTIFACT_HASH"}


def test_canonical_result_is_deterministic(benchmark) -> None:
    artifacts = benchmark.artifacts(PIPELINE)
    item = fault("E6_MODEL_EXPLAINER_VERSION")
    assert evaluate(artifacts, item, "O_FUZZYXAI").canonical_sha256 == evaluate(artifacts, item, "O_FUZZYXAI").canonical_sha256


def test_unregistered_mutation_is_not_constructible() -> None:
    assert all(item.case_id.startswith(("C", "E")) for item in FAULTS)
    assert len({item.case_id for item in FAULTS}) == 10


def test_adapters_do_not_change_external_source(benchmark) -> None:
    artifacts = benchmark.artifacts(PIPELINE)
    before = {path: path.read_bytes() for path in artifacts.root.glob("*") if path.is_file()}
    benchmark.artifacts(PIPELINE)
    assert all(path.read_bytes() == value for path, value in before.items())


def test_all_four_pipelines_share_same_fault_contracts(benchmark) -> None:
    for item in FAULTS[2:]:
        observed = {
            evaluate(benchmark.artifacts(pipeline), item, "O_FUZZYXAI").contract_id
            for pipeline in (
                "ext1-sklearn-column-transformer",
                "ext2-shap-tree-explainer",
                "ext3-mlflow-elasticnet",
                "ext4-lime-tabular",
            )
        }
        assert observed == {item.contract_id}
