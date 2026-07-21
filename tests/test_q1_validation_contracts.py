from __future__ import annotations

import json
from pathlib import Path

import pytest

from fuzzyxai.q1_validation import (
    CascadePolicy,
    CascadeSignals,
    ClaimStatus,
    EvidenceOrigin,
    ExternalGate,
    FidelityPair,
    HypothesisResult,
    OperationKind,
    PartitionRole,
    Q1CalibrationConfig,
    SplitUseRecord,
    noninferiority_test,
)
from fuzzyxai.q1_validation.critical_rupture import (
    StructuralDefect,
    StructuralObservation,
    diagnose_structural_ruptures,
    structural_metrics,
)
from fuzzyxai.q1_validation.local_explainers import LocalExplanation, wrap_local_explanation
from fuzzyxai.q1_validation.real_benchmarks import _read_ts
from fuzzyxai.q1_validation.rule_ablation import AblationPair, RuleDescriptor, select_matched_random_rule, summarize_ablation
from fuzzyxai.q1_validation.traceability import EvidenceClaim, MissingnessPrediction, evaluate_missingness, traceability_score


def test_test_partition_is_rejected_for_fit_selection_and_calibration() -> None:
    for operation in (OperationKind.FIT, OperationKind.SELECT, OperationKind.CALIBRATE):
        with pytest.raises(ValueError, match="test partition"):
            SplitUseRecord("leak", operation, (PartitionRole.TEST,), {"test": "sha"})
    record = SplitUseRecord("held-out", OperationKind.EVALUATE, (PartitionRole.TEST,), {"test": "sha"})
    assert record.to_dict()["partitions"] == ["test"]


def test_noninferiority_uses_lower_confidence_bound_and_preserves_failures() -> None:
    passing = [FidelityPair(str(index), "SHAP", 0.4, 0.395, True, True, True, True) for index in range(30)]
    result = noninferiority_test(passing, bootstrap_repetitions=300)
    assert result.noninferior
    assert result.lower_bound >= -0.02
    failing = [FidelityPair(str(index), "SHAP", 0.4, 0.35, True, True, True, True) for index in range(30)]
    assert not noninferiority_test(failing, bootstrap_repetitions=300).noninferior


def test_wrapping_adds_traceability_without_changing_attribution() -> None:
    local = LocalExplanation(
        method="SHAP",
        object_id="85",
        attribution=(0.2, -0.1),
        feature_names=("a", "b"),
        model_output=0.7,
        background_hash="background",
        budget=100,
        seed=42,
        provenance={"model_sha256": "m", "data_sha256": "d", "method_version": "1"},
    )
    wrapped = wrap_local_explanation(local, evidence_refs=("evidence:85",))
    assert wrapped.attribution == local.attribution
    assert wrapped.traceability == 1.0


def test_traceability_and_missingness_are_measured_separately() -> None:
    claims = [
        EvidenceClaim("c1", ("e1",), ("dataset",), ("v1",), ("sha",)),
        EvidenceClaim("c2", ("e2",), (), ("v1",), ("sha",)),
    ]
    assert traceability_score(claims) == 0.5
    report = evaluate_missingness(
        [
            MissingnessPrediction("1", ("history",), ("history",), False),
            MissingnessPrediction("2", (), (), True),
            MissingnessPrediction("3", ("hash",), ("hash",), False),
        ]
    )
    assert report.f1 == 1.0
    assert report.false_certification_rate == 0.0


def test_adaptive_cascade_escalates_missing_conflicting_and_rare_objects() -> None:
    policy = CascadePolicy()
    safe = CascadeSignals(0.95, True, 0.01, 0.95, 0.01, False, 0.8)
    unstable = CascadeSignals(0.70, True, 0.1, 0.65, 0.1, False, 0.4)
    rare = CascadeSignals(0.95, True, 0.01, 0.95, 0.01, True, 0.8)
    assert policy.level(safe).value == "A"
    assert policy.level(unstable).value == "B"
    assert policy.level(rare).value == "C"


def test_structural_rupture_requires_typed_evidence_and_reports_exact_type() -> None:
    observation = StructuralObservation(
        object_id="85",
        available_evidence=frozenset({"model"}),
        required_evidence=frozenset({"model", "history"}),
        provenance_valid=True,
        forbidden_conflict=False,
        representation_covered=True,
        reduction_loss=0.0,
        explanation_stability=1.0,
        distribution_shift=0.0,
        cross_model_disagreement=0.0,
        evidence_refs={StructuralDefect.MISSING_REQUIRED_EVIDENCE: ("trace:history",)},
    )
    diagnosis = diagnose_structural_ruptures(observation)
    assert diagnosis.defects == (StructuralDefect.MISSING_REQUIRED_EVIDENCE,)
    metrics = structural_metrics([(StructuralDefect.MISSING_REQUIRED_EVIDENCE,)], [diagnosis])
    assert metrics["f1"] == 1.0
    assert metrics["false_certification_rate"] == 0.0


def test_rule_ablation_requires_matched_baseline_and_fifty_pairs() -> None:
    selected = RuleDescriptor("R1", 0.1, 0.8, 0.9, 0.1, 0.11, 3, 0.8, "1")
    candidates = (
        selected,
        RuleDescriptor("R2", 0.11, 0.2, 0.1, 0.5, 0.10, 3, 0.6, "1"),
        RuleDescriptor("R3", 0.5, 0.2, 0.1, 0.5, 0.50, 8, 0.6, "1"),
    )
    assert select_matched_random_rule(selected, candidates).rule_id == "R2"
    with pytest.raises(ValueError, match="50"):
        summarize_ablation((AblationPair(0, 0, "R1", "R2", 0.2, 0.1, "S4"),))
    pairs = tuple(AblationPair(index % 10, index // 10, "R1", "R2", 0.2, 0.1, "S4") for index in range(50))
    assert summarize_ablation(pairs)["n_pairs"] == 50


def test_external_gate_and_supported_claim_fail_closed() -> None:
    with pytest.raises(ValueError, match="participant"):
        ExternalGate("comprehension", "completed")
    with pytest.raises(ValueError, match="external placeholders"):
        HypothesisResult(
            "H7",
            ClaimStatus.SUPPORTED,
            {},
            ("planned.json",),
            (),
            "useful",
            "validated",
            EvidenceOrigin.EXTERNAL,
        )


def test_q1_calibration_config_validates_thresholds() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        Q1CalibrationConfig(1.1, 0.5, 1, 1, 1, 1, 1, 0.2, 0.3, 10, 5, 1, 2)


def test_electric_devices_reader_accepts_an_official_split_below_10k(tmp_path: Path) -> None:
    split = tmp_path / "ElectricDevices_TRAIN.ts"
    split.write_text("@problemName ElectricDevices\n@data\n1.0,2.0,?:3\n", encoding="utf-8")
    values, labels = _read_ts(split)
    assert values.tolist() == [[1.0, 2.0, 0.0]]
    assert labels.tolist() == [3.0]


def test_packaged_external_gates_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    gates = json.loads((root / "research/preregistration/q1_external_gates.json").read_text(encoding="utf-8"))
    assert gates["stable_release_allowed"] is False
    studies = [details for details in gates.values() if isinstance(details, dict)]
    assert studies
    assert all(not details["claim_allowed"] for details in studies)
