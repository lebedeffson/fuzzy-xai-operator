from __future__ import annotations

from fuzzyxai.ml_vertical.pipeline import CONTRACT_STAGE

PIPELINE = "breast-cancer-logreg-linearshap"


def test_local_violation_does_not_require_route_graph(service) -> None:
    result = service.mutate(PIPELINE, "TRANSFORM_FINITE", "L1", "B_LOCAL_STRONG")
    assert result.detected and result.contract_id == "FINITE_TRANSFORMED_VALUES"
    assert result.diagnostic_cut is None


def test_cross_stage_violation_is_not_visible_to_local_baseline(service) -> None:
    result = service.mutate(PIPELINE, "FIT_SCOPE", "L2", "B_LOCAL_STRONG")
    assert not result.detected
    assert result.pipeline_status == "VALID"


def test_pairwise_rules_do_not_use_global_cut(service) -> None:
    result = service.mutate(PIPELINE, "FEATURE_SCHEMA_CASCADE", "L1", "B_PAIRWISE_RULES")
    assert result.detected
    assert result.diagnostic_cut is None
    assert result.root_cause is None
    assert result.reported_symptom_count > 1


def test_fuzzyxai_selects_single_cascade_root(service) -> None:
    result = service.mutate(PIPELINE, "FEATURE_SCHEMA_CASCADE", "L1")
    assert result.root_cause == "FEATURE_ORDER"
    assert result.diagnostic_cut == {"contracts": ["FEATURE_ORDER"], "size": 1, "solver": "causal_route_minimal_cut"}
    assert "MODEL_INPUT_SCHEMA" in result.dependent_violations


def test_repair_removes_cascade_symptoms_and_recertifies(service) -> None:
    result = service.mutate(PIPELINE, "FEATURE_SCHEMA_CASCADE", "L1")
    assert result.repair_executed
    assert result.target_contract_repaired
    assert result.recertified
    assert result.new_critical_violations == 0


def test_full_recertification_rechecks_all_contracts(service) -> None:
    artifacts = service.prepare(PIPELINE)
    recertification = service.mutate(PIPELINE, "MODEL_ARTIFACT", "L1")
    assert recertification.recertified
    assert len(artifacts.registration.supported_contracts) == len(CONTRACT_STAGE) == 28


def test_repair_has_verified_rollback(service) -> None:
    result = service.mutate(PIPELINE, "EXPLAINER_BINDING", "L2")
    assert result.repair_plan and result.repair_plan["rollback"]
    assert result.rollback_verified


def test_greedy_mode_does_not_claim_graph_root(service) -> None:
    result = service.mutate(PIPELINE, "FEATURE_SCHEMA_CASCADE", "L2", "B_GREEDY_CROSS_STAGE")
    assert result.root_cause == result.contract_id
    assert result.diagnostic_cut is None
    assert not result.repair_executed


def test_canonical_result_excludes_runtime_jitter(service) -> None:
    first = service.mutate(PIPELINE, "SHAP_CONSISTENCY", "L1")
    second = service.mutate(PIPELINE, "SHAP_CONSISTENCY", "L1")
    assert first.canonical_sha256 == second.canonical_sha256


def test_false_certification_definition_is_triggered_for_local_miss(service) -> None:
    result = service.mutate(PIPELINE, "FIT_SCOPE", "L1", "B_LOCAL_STRONG")
    assert result.pipeline_status == "VALID"
    assert result.action == "ACCEPT"
