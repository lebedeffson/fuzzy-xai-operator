from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.real_training_experiment.run_empirical_validation import run
from fuzzyxai.evidence import (
    CounterfactualEvidence,
    comparison_statement,
    validate_domain_language,
)


def test_small_reference_uses_rank_not_percentile_claim() -> None:
    statement = comparison_statement(
        [0.1, 0.2, 0.3, 0.4],
        0.5,
        reference_label="контрольная выборка",
        representation="исходные признаки",
    )
    assert statement.wording_policy == "small_sample_rank"
    assert "максимальным среди 4" in statement.text
    assert "%" not in statement.text


def test_domain_semantic_validator_rejects_opposite_direction() -> None:
    result = validate_domain_language(
        {
            "version": "test-v1",
            "features": {
                "feature_a": {
                    "label": "показатель A",
                    "meaning": "контрольный показатель",
                    "expected_direction": "increases_target",
                    "expert_review_status": "not_reviewed",
                }
            },
        },
        contribution_directions={"feature_a": -0.4},
    )
    assert result.status == "rejected"
    assert "contradicts" in result.errors[0]


def test_actionable_counterfactual_requires_domain_validation() -> None:
    with pytest.raises(ValueError, match="actionable=True"):
        CounterfactualEvidence(
            source_prediction=0,
            target_prediction=1,
            changed_features={"feature": {"from": 1.0, "to": 2.0}},
            changed_regions=(),
            changed_rules=(),
            minimality=0.1,
            plausibility=0.9,
            stability=None,
            expected_effect=None,
            observed_effect=0.2,
            actionability="not reviewed",
            limitations=("domain feasibility unavailable",),
            mode="actionable_counterfactual",
            actionable=None,
        )


def test_real_training_run_is_measured_and_reproducible(tmp_path: Path) -> None:
    first = run(tmp_path / "first")
    second = run(tmp_path / "second")
    assert first["result_origin"] == "measured"
    assert first["checkpoints"] == 30
    assert first["checkpoint_hashes_unique"] == 30
    assert first["selected_case"]["public_id"] == "case_real_001"
    assert first["selected_case"]["forgetting_events"]
    assert first["subgroup"]["subgroup_definition_hash"] == second["subgroup"]["subgroup_definition_hash"]
    assert first["selected_case"] == second["selected_case"]
    assert first["rule_ablation"] == second["rule_ablation"]
    assert set(first["similar_case_roles"]) == {"support", "counterexample"}
    assert first["counterfactual_modes"] == ["sensitivity_analysis"]
    assert first["release_gate"] == "blocked_external_pilot_and_domain_review"
    cross_model = json.loads((tmp_path / "first/cross_model_matrix.json").read_text(encoding="utf-8"))
    by_model = {row["model"]: row for row in cross_model}
    assert by_model["black_box_callable"]["native_rule_count"] == 0
    assert by_model["decision_tree"]["native_rule_count"] > 0
    assert by_model["sugeno_native_rules"]["native_rule_count"] > 0
    assert all(not row["graph_errors"] for row in cross_model)
