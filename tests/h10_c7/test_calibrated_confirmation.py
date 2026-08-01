from __future__ import annotations

import inspect
from dataclasses import asdict
from pathlib import Path

import pytest
from fuzzyxai.experiments.h10_c7_confirmation import (
    _apply_active_probability,
    _select_threshold,
)
from fuzzyxai.repository_diagnostics.calibrated_confirmation import (
    FEATURE_NAMES,
    FORBIDDEN_FEATURE_NAMES,
    CalibratedDiagnosisConfirmer,
    ConfirmationFeatures,
    DeterministicConfirmationModel,
    LogisticConfirmationModel,
    extract_confirmation_features,
    feature_vector,
)
from fuzzyxai.repository_diagnostics.contract_inference_v2 import (
    ContractPrediction,
)
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedCandidate,
    GuidedDiagnosis,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import IncidentQuery
from fuzzyxai.repository_diagnostics.incident_router import RoutingDecision


def _features(**updates: object) -> ConfirmationFeatures:
    values: dict[str, object] = {
        "candidate_rank": 1,
        "candidate_score": 3.0,
        "candidate_margin": 1.0,
        "candidate_margin_normalized": 0.2,
        "contract_score": 0.9,
        "contract_margin": 0.5,
        "contract_margin_normalized": 0.4,
        "joint_score": 3.9,
        "joint_margin": 1.2,
        "joint_margin_normalized": 0.2,
        "independent_evidence_groups": 4,
        "runtime_evidence_groups": 2,
        "retrieval_channels": 2,
        "has_assertion_evidence": True,
        "has_exception_evidence": False,
        "has_traceback_evidence": True,
        "has_executed_slice_evidence": True,
        "has_dynamic_call_evidence": False,
        "has_exact_symbol_evidence": True,
        "has_repograph_evidence": False,
        "has_lexical_retrieval_evidence": True,
        "has_legacy_retrieval_evidence": False,
        "has_contract_direct_evidence": True,
        "has_active_probe_evidence": False,
        "production_symbol": True,
        "test_symbol": False,
        "rank_stability": 0.8,
        "contract_stability": 0.8,
    }
    values.update(updates)
    return ConfirmationFeatures(**values)


def _candidate(
    *,
    path: str = "src/loader.py",
    symbol: str = "load_schema",
    sources: tuple[str, ...] = (
        "bm25",
        "exact_symbol",
        "executed_slice",
        "traceback",
    ),
    evidence: tuple[str, ...] = (
        "direct_observation:attributeerror",
        "observed:attributeerror",
    ),
    family: str = "API_BEHAVIOR",
    score: float = 3.0,
) -> GuidedCandidate:
    contract = ContractPrediction(family, "CONFIGURATION", 0.9, evidence)
    alternative = ContractPrediction(
        "DATA_SCHEMA",
        "DATA",
        0.3,
        ("observed:schema",),
    )
    return GuidedCandidate(
        f"{path}::{symbol}",
        path,
        symbol,
        score,
        contract,
        (contract, alternative),
        sources,
        10,
        ("failure",),
    )


def _diagnosis(candidate: GuidedCandidate) -> GuidedDiagnosis:
    second = _candidate(
        path="src/other.py",
        symbol="other",
        sources=("repograph",),
        score=1.0,
    )
    return GuidedDiagnosis(
        "R5",
        "DIAGNOSIS_CANDIDATES",
        (candidate, second),
        (),
        RoutingDecision("EXECUTED_SLICE_GRAPH", ("traceback_available",)),
    )


def _query() -> IncidentQuery:
    return IncidentQuery(
        "opaque-id",
        "AttributeError while loading schema",
        ("tests/test_loader.py::test_schema",),
        "AttributeError in load_schema",
        "expected schema but received invalid value",
    )


def test_duplicate_exception_records_form_one_evidence_group() -> None:
    one = extract_confirmation_features(_diagnosis(_candidate()), _query())
    duplicate = extract_confirmation_features(
        _diagnosis(
            _candidate(
                evidence=(
                    "direct_observation:attributeerror",
                    "observed:attributeerror",
                    "observed:attributeerror",
                )
            )
        ),
        _query(),
    )
    assert one.independent_evidence_groups == duplicate.independent_evidence_groups


def test_traceback_and_executed_slice_are_independent_groups() -> None:
    both = extract_confirmation_features(_diagnosis(_candidate()), _query())
    traceback_only = extract_confirmation_features(
        _diagnosis(_candidate(sources=("traceback",))),
        _query(),
    )
    assert both.independent_evidence_groups == (
        traceback_only.independent_evidence_groups + 3
    )


def test_bm25_and_exact_symbol_are_distinct_channels() -> None:
    result = extract_confirmation_features(_diagnosis(_candidate()), _query())
    assert result.has_lexical_retrieval_evidence
    assert result.has_exact_symbol_evidence
    assert result.retrieval_channels == 2


def test_gold_fields_are_not_model_features() -> None:
    assert not set(FEATURE_NAMES).intersection(FORBIDDEN_FEATURE_NAMES)
    assert not set(asdict(_features())).intersection(
        {"gold_file", "gold_symbol", "gold_contract", "gold_patch"}
    )


def test_repository_name_is_not_model_feature() -> None:
    assert "repository" not in FEATURE_NAMES
    assert "repository_name" not in asdict(_features())


def test_incident_id_is_not_model_feature() -> None:
    assert "incident_id" not in FEATURE_NAMES
    assert "incident_id" not in asdict(_features())


def test_training_scaler_does_not_observe_test_features() -> None:
    train = (_features(candidate_margin_normalized=0.1),) * 2
    model = LogisticConfirmationModel().fit(train, (0, 1))
    expected = feature_vector(train[0])[0]
    assert model.scaler.mean_[0] == expected
    assert model.scaler.mean_[0] != 99.0


def test_threshold_selector_accepts_training_rows_only() -> None:
    signature = inspect.signature(_select_threshold)
    assert "train" in signature.parameters
    assert "test" not in signature.parameters
    assert "repository" not in signature.parameters


def test_loro_artifact_uses_out_of_fold_predictions_only() -> None:
    status = Path("results/h10_c7/confirmation/R5C_STATUS.json")
    if not status.exists():
        pytest.skip("R5C recorded replay has not been generated")
    import json

    value = json.loads(status.read_text(encoding="utf-8"))
    assert value["leakage_audit"]["checks"][
        "scoring_uses_only_oof_predictions"
    ]
    assert value["leakage_audit"]["checks"]["every_incident_predicted_once"]


def test_unknown_contract_is_never_confirmed() -> None:
    decision = CalibratedDiagnosisConfirmer().decide(
        features=_features(),
        contract_family="UNKNOWN_CONTRACT",
        probability=0.99,
        threshold=0.5,
    )
    assert decision.status != "DIAGNOSIS_CONFIRMED"
    assert "unknown_contract" in decision.rejected_reasons


def test_test_symbol_is_not_confirmed() -> None:
    decision = CalibratedDiagnosisConfirmer().decide(
        features=_features(production_symbol=False, test_symbol=True),
        contract_family="DATA_SCHEMA",
        probability=0.99,
        threshold=0.5,
    )
    assert decision.status == "DIAGNOSIS_PROBABLE"
    assert "test_or_service_symbol" in decision.rejected_reasons


def test_high_score_without_independent_evidence_is_not_confirmed() -> None:
    decision = CalibratedDiagnosisConfirmer().decide(
        features=_features(
            candidate_score=100.0,
            independent_evidence_groups=1,
            runtime_evidence_groups=0,
            has_assertion_evidence=False,
            has_traceback_evidence=False,
            has_executed_slice_evidence=False,
        ),
        contract_family="DATA_SCHEMA",
        probability=0.99,
        threshold=0.5,
    )
    assert decision.status != "DIAGNOSIS_CONFIRMED"


def test_stable_top_candidate_with_three_groups_can_be_confirmed() -> None:
    decision = CalibratedDiagnosisConfirmer().decide(
        features=_features(
            independent_evidence_groups=3,
            rank_stability=0.8,
        ),
        contract_family="DATA_SCHEMA",
        probability=0.9,
        threshold=0.8,
    )
    assert decision.status == "DIAGNOSIS_CONFIRMED"


def test_active_evidence_changes_confirmation_features() -> None:
    before = extract_confirmation_features(_diagnosis(_candidate()), _query())
    after = extract_confirmation_features(
        _diagnosis(_candidate()),
        _query(),
        active_probe=True,
    )
    assert not before.has_active_probe_evidence
    assert after.has_active_probe_evidence
    assert after.independent_evidence_groups == (
        before.independent_evidence_groups + 1
    )


def test_active_evidence_without_probability_gain_is_not_applied() -> None:
    class RegressiveModel:
        model_id = "fixture"

        @staticmethod
        def predict_probability(features: ConfirmationFeatures) -> float:
            return 0.6 if features.has_active_probe_evidence else 0.7

    replay = type(
        "Replay",
        (),
        {
            "active_features": _features(has_active_probe_evidence=True),
        },
    )()
    probability, status = _apply_active_probability(
        RegressiveModel(),
        replay,
        0.7,
        0.8,
    )
    assert probability == 0.7
    assert status == "ACTIVE_EVIDENCE_NO_POSITIVE_GAIN"


def test_recorded_r5_top_k_is_unchanged() -> None:
    status = Path("results/h10_c7/confirmation/R5C_STATUS.json")
    if not status.exists():
        pytest.skip("R5C recorded replay has not been generated")
    import json

    value = json.loads(status.read_text(encoding="utf-8"))
    checks = value["r5_retrieval_immutability"]["checks"]
    assert checks["top_10_signatures_unchanged"]
    assert checks["recall_at_10_unchanged"]
    assert checks["recall_at_20_unchanged"]


def test_c0_is_reproducible() -> None:
    model = DeterministicConfirmationModel()
    first = model.predict_probability(_features())
    second = model.predict_probability(_features())
    assert first == second


def test_c1_is_reproducible_with_fixed_seed() -> None:
    features = (
        _features(candidate_margin_normalized=0.1),
        _features(candidate_margin_normalized=0.2),
        _features(candidate_margin_normalized=0.3),
        _features(candidate_margin_normalized=0.4),
    )
    labels = (0, 0, 1, 1)
    first = LogisticConfirmationModel(seed=1707).fit(features, labels)
    second = LogisticConfirmationModel(seed=1707).fit(features, labels)
    assert first.predict_probability(features[0]) == (
        second.predict_probability(features[0])
    )
    assert first.parameters() == second.parameters()


def test_rules_contain_no_specific_repository_names() -> None:
    source = Path(
        "framework/fuzzyxai/fuzzyxai/repository_diagnostics/"
        "calibrated_confirmation.py"
    ).read_text(encoding="utf-8").lower()
    for name in ("fastapi", "pysnooper", "youtube-dl", "httpie", "tornado"):
        assert name not in source


def test_rules_contain_no_specific_gold_symbols() -> None:
    source = Path(
        "framework/fuzzyxai/fuzzyxai/repository_diagnostics/"
        "calibrated_confirmation.py"
    ).read_text(encoding="utf-8")
    for symbol in (
        "create_cloned_field",
        "jsonable_encoder",
        "format_file_in_place",
        "build_format_selector",
    ):
        assert symbol not in source
