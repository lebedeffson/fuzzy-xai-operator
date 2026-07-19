from __future__ import annotations

import numpy as np
import pytest

from fuzzyxai.experiments import (
    CriticalRuptureType,
    ExperimentGate,
    ExperimentRunManifest,
    detect_critical_ruptures,
    holm_adjust,
    mcnemar_exact,
    paired_summary,
)
from fuzzyxai.experiments.datasets import build_all_controlled
from fuzzyxai.experiments.policies import PolicySignals, apply_policy
from fuzzyxai.experiments.uncertainty_selection import evaluate_selection_modes


def test_pass_gate_rejects_planned_or_failed_evidence() -> None:
    with pytest.raises(ValueError, match="PASS requires measured"):
        ExperimentGate("E1", "PASS", "planned_not_run", {"objects": True})
    with pytest.raises(ValueError, match="every declared check"):
        ExperimentGate("E1", "PASS", "measured", {"objects": False})


def test_release_manifest_remains_blocked_by_external_gates() -> None:
    gate = ExperimentGate("E1", "PASS", "controlled", {"objects": True})
    manifest = ExperimentRunManifest(
        schema_version="1.0",
        profile="smoke",
        commit="abc",
        branch="test",
        seed=42,
        threads=1,
        experiments=(gate,),
        external_gates={"expert_review": "planned_not_run"},
    )
    assert manifest.release_status == "BLOCKED"
    assert manifest.to_dict()["tag_allowed"] is False


def test_all_controlled_modalities_are_aligned() -> None:
    datasets = build_all_controlled(n_objects=120)
    assert {item.modality for item in datasets} == {"tabular", "image", "text", "time_series"}
    assert all(item.n_objects == 120 for item in datasets)
    assert all(len(item.labels) == len(item.critical_mask) for item in datasets)
    assert all(item.metadata["source_type"] == "controlled_synthetic" for item in datasets)


def test_critical_rupture_requires_evidence_and_is_typed() -> None:
    with pytest.raises(ValueError, match="missing evidence refs"):
        detect_critical_ruptures(
            object_id="object-1",
            required_evidence_present=False,
            forbidden_rule_conflict=False,
            provenance_verified=True,
            representation_covers_profile=True,
            reduction_loss=0.0,
            reduction_loss_threshold=0.2,
            distribution_shift=0.0,
            distribution_shift_threshold=0.2,
            explanation_stability=1.0,
            stability_threshold=0.5,
            cross_model_disagreement=0.0,
            disagreement_threshold=0.2,
            evidence_refs={},
        )
    ruptures = detect_critical_ruptures(
        object_id="object-1",
        required_evidence_present=False,
        forbidden_rule_conflict=False,
        provenance_verified=True,
        representation_covers_profile=True,
        reduction_loss=0.0,
        reduction_loss_threshold=0.2,
        distribution_shift=0.0,
        distribution_shift_threshold=0.2,
        explanation_stability=1.0,
        stability_threshold=0.5,
        cross_model_disagreement=0.0,
        disagreement_threshold=0.2,
        evidence_refs={"missing_required_evidence": ("evidence:1",)},
    )
    assert ruptures[0].rupture_type is CriticalRuptureType.MISSING_REQUIRED_EVIDENCE


def test_pairwise_statistics_and_holm_are_deterministic() -> None:
    report = paired_summary([0.8, 0.7, 0.9], [0.7, 0.65, 0.85], seed=9)
    assert report.n_pairs == 3
    assert report.worsening_fraction == 1.0
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert mcnemar_exact([True, False, True], [False, True, True])["discordant_pairs"] == 2


def test_full_policy_blocks_rupture_and_history_limits_acceptance() -> None:
    signals = PolicySignals(
        confidence=np.asarray([0.9, 0.9, 0.9]),
        shap_support=np.asarray([0.8, 0.8, 0.8]),
        lime_support=np.asarray([0.8, 0.8, 0.8]),
        explanation_stability=np.asarray([0.9, 0.9, 0.9]),
        critical_rupture=np.asarray([False, True, False]),
        history_instability=np.asarray([False, False, True]),
    )
    assert apply_policy("P5", signals).tolist() == ["accept", "block", "review"]


def test_hierarchy_claim_is_blocked_when_adaptive_uses_fml_above_90_percent() -> None:
    profiles = [("distribution_shift",)] * 19 + [("aleatoric",)]
    risks = {name: [0.1] * len(profiles) for name in ("F0", "Fint", "NAS", "FML", "adaptive")}
    result = evaluate_selection_modes(profiles, epsilon=0.01, action_risks=risks)
    assert result["adaptive_fml_fraction"] == pytest.approx(0.95)
    assert result["practical_hierarchy_claim_allowed"] is False
