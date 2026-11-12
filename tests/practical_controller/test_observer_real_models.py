"""End-to-end seeded observer exercise with fitted models and held-out inputs."""

from __future__ import annotations

from experiments.observer_local_evidence_benchmark import run_benchmark


def test_observer_contract_detects_declared_faults_on_three_fitted_models() -> None:
    report = run_benchmark(n_per_dataset=5, seed=42)
    assert report["n_rows"] == 3 * 5 * 16
    assert report["maximum_clean_logit_reconstruction_error"] < 1e-8

    families = report["by_family"]
    assert families["clean"]["accepted_structurally"] == 15
    assert families["clean"]["hash_verified"] == 15
    assert families["clean"]["structural_plus_sum_accepted"] == 15
    assert families["clean"]["blocked"] == 0
    assert families["clean"]["review_required"] == 0

    for name in (
        "object_substitution",
        "target_substitution",
        "feature_order_substitution",
        "flattened_multiclass_attribution",
        "global_attribution_as_local",
        "source_digest_substitution",
    ):
        assert families[name]["blocked"] == 15, name
        assert families[name]["accepted_structurally"] == 0, name

    assert families["missing_target"]["review_required"] == 15
    # This negative result is intentional: a shape/identity contract cannot
    # certify the semantics of the attribution values it was handed.
    assert families["silent_value_permutation"]["blocked"] == 15
    assert families["silent_value_permutation"]["structural_only_accepted"] == 15
    assert families["silent_value_permutation"]["structural_plus_sum_accepted"] == 15
    assert families["silent_value_permutation"]["hash_verified"] == 15

    for delta in (1e-4, 1e-2, 0.1):
        case = families[f"randomized_zero_sum_{delta:g}"]
        assert case["n"] == 15
        assert case["blocked"] == 15
        assert case["hash_verified"] == 15
        assert case["structural_only_accepted"] == 15
        assert case["structural_plus_sum_accepted"] == 15

    assert all(report["by_dataset"][name]["clean"]["n"] == 5 for name in ("breast_cancer", "wine", "digits"))
    assert families["source_digest_substitution"]["hash_verified"] == 0
    assert families["stale_input_same_object"]["blocked"] == 15
    assert families["stale_input_same_object"]["hash_verified"] == 15
    assert families["stale_input_same_object"]["identity_baseline_accepted"] == 15
    assert families["missing_input_binding"]["review_required"] == 15
    assert families["missing_input_binding"]["hash_verified"] == 15
    assert families["wrong_reference_identity"]["blocked"] == 15
    assert families["wrong_reference_identity"]["hash_verified"] == 15
    assert families["missing_reference_binding"]["review_required"] == 15
    assert families["missing_reference_binding"]["hash_verified"] == 15
    assert report["claim_allowed"] is False



def test_exploratory_cross_seed_stability_and_model_fingerprints() -> None:
    # This checks deterministic implementation stability, not independent
    # confirmatory validation; the same three datasets are reused.
    first = run_benchmark(n_per_dataset=3, seed=11)
    second = run_benchmark(n_per_dataset=3, seed=27)
    assert first["n_rows"] == second["n_rows"] == 3 * 3 * 16
    assert first["model_fingerprints"] != second["model_fingerprints"]
    assert set(first["model_fingerprints"]) == {"breast_cancer", "wine", "digits"}
    assert all(len(value) == 64 for value in first["model_fingerprints"].values())
    for report in (first, second):
        assert report["maximum_clean_logit_reconstruction_error"] < 1e-8
        assert report["by_family"]["clean"]["accepted_structurally"] == 9
        assert report["by_family"]["stale_input_same_object"]["blocked"] == 9
        for family in (
            "silent_value_permutation",
            "randomized_zero_sum_0.0001",
            "randomized_zero_sum_0.01",
            "randomized_zero_sum_0.1",
        ):
            assert report["by_family"][family]["blocked"] == 9
            assert report["by_family"][family]["structural_plus_sum_accepted"] == 9
