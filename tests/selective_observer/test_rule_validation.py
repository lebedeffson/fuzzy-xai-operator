from __future__ import annotations

import numpy as np

from fuzzyxai.selective_observer import (
    ConfirmatoryProtocolLock,
    ResearchPartition,
    RuleAblationObservation,
    RuleProfile,
    evaluate_confirmatory_ablation,
    evaluate_planted_rule_recovery,
    inject_planted_rule,
    matched_controls,
)


def _profile(rule_id: str, score: float, redundancy: float) -> RuleProfile:
    return RuleProfile(
        rule_id=rule_id,
        subgroup_specificity=score,
        subgroup_coverage=score,
        activation_stability=score,
        functional_redundancy=redundancy,
        unique_prediction_fraction=score,
        margin_contribution=score,
        depth=3,
        length=3,
        selection_partition=ResearchPartition.VALIDATION,
    )


def test_formative_planted_rule_is_ranked_without_real_data_claim() -> None:
    profiles = [_profile("planted", 0.95, 0.02), *[_profile(f"rule-{index}", 0.4, 0.5) for index in range(8)]]
    result = evaluate_planted_rule_recovery(profiles, ["planted"], top_k=1)
    assert result["planted_rule_recall"] == 1.0
    assert result["methodology_supported"] is True
    assert result["real_data_claim_allowed"] is False
    assert len(matched_controls(profiles[0], profiles)) == 5


def test_semisynthetic_injection_records_scope() -> None:
    values = np.arange(200, dtype=float).reshape(100, 2)
    labels = np.zeros(100, dtype=int)
    changed, mask, metadata = inject_planted_rule(values, labels, feature=0, target_class=1)
    assert mask.any()
    assert np.all(changed[mask] == 1)
    assert metadata["phase"] == "formative_semisynthetic"
    assert metadata["confirmatory_claim_allowed"] is False


def test_confirmatory_specific_effect_requires_two_heldout_datasets(protocol_lock: ConfirmatoryProtocolLock) -> None:
    observations = [
        RuleAblationObservation(
            dataset_id=dataset,
            candidate_rule_id=f"candidate-{dataset}-{index}",
            candidate_effect=0.10 + 0.005 * index,
            matched_control_effects=(0.01, 0.02, 0.015, 0.0, 0.01),
        )
        for dataset in ("new-a", "new-b")
        for index in range(10)
    ]
    result = evaluate_confirmatory_ablation(observations, protocol_lock, bootstrap_repetitions=100)
    assert result["status"] == "supported"
    assert result["claim_allowed"] is True
    assert result["datasets"] == ["new-a", "new-b"]
