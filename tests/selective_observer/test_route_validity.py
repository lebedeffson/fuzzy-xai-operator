from __future__ import annotations

from fuzzyxai.selective_observer import (
    ConfirmatoryProtocolLock,
    PredictiveRouteExample,
    ResearchPartition,
    RouteContractRecord,
    SelectiveAction,
    evaluate_action_contract,
    evaluate_predictive_increment,
)


def test_action_contract_targets_invalid_certification_not_model_error() -> None:
    records = [
        RouteContractRecord(
            object_id=f"fault-{index}",
            mandatory_nodes=("evidence", "claim", "diagnostic", "action"),
            observed_nodes=("evidence", "claim", "action"),
            fault_type="missing_diagnostic",
            fault_source="diagnostic",
            detected_fault_type="missing_diagnostic",
            detected_fault_source="diagnostic",
            rupture_severity=0.9,
            requested_action=SelectiveAction.FULL_REVIEW,
            model_error=bool(index % 2),
        )
        for index in range(30)
    ]
    records.extend(
        RouteContractRecord(
            object_id=f"clean-{index}",
            mandatory_nodes=("evidence", "claim", "diagnostic", "action"),
            observed_nodes=("evidence", "claim", "diagnostic", "action"),
            fault_type=None,
            fault_source=None,
            detected_fault_type=None,
            detected_fault_source=None,
            rupture_severity=0.0,
            requested_action=SelectiveAction.ACCEPT,
            model_error=bool(index % 3 == 0),
        )
        for index in range(30)
    )
    result = evaluate_action_contract(records)
    assert result["structural_claim_allowed"] is True
    assert result["false_certification_rate"] == 0.0
    assert result["model_error_prediction_claim_allowed"] is False


def test_predictive_increment_uses_frozen_test_only_for_evaluation(protocol_lock: ConfirmatoryProtocolLock) -> None:
    development = []
    test = []
    for index in range(120):
        error = index % 4 == 0
        item = PredictiveRouteExample(
            object_id=f"dev-{index}",
            baseline_features=(0.4 + 0.1 * (index % 3), 0.2, 0.1),
            typed_route_features=(0.95 if error else 0.05, 0.8 if error else 0.1),
            model_error=error,
            partition=ResearchPartition.TRAIN if index % 2 else ResearchPartition.VALIDATION,
            source_features_are_oof=True,
        )
        development.append(item)
    for index in range(60):
        error = index % 4 == 0
        test.append(
            PredictiveRouteExample(
                object_id=f"test-{index}",
                baseline_features=(0.4 + 0.1 * (index % 3), 0.2, 0.1),
                typed_route_features=(0.95 if error else 0.05, 0.8 if error else 0.1),
                model_error=error,
                partition=ResearchPartition.TEST,
                source_features_are_oof=False,
            )
        )
    result = evaluate_predictive_increment(development, test, protocol_lock, bootstrap_repetitions=100)
    assert result["m1_auprc"] > result["m0_auprc"]
    assert result["test_opened_once"] is True
    assert result["structural_h5a_unchanged"] is True
