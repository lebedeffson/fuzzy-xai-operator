"""Protocol-locked postdevelopment target-pairing benchmark regression."""

from __future__ import annotations

from experiments.ml_target_witness_postdev_benchmark import run_benchmark


def test_target_consistency_benchmark_uses_declared_models_and_negative_controls() -> None:
    result = run_benchmark()
    assert result["n_datasets"] == 3
    assert result["n_cases"] == 18
    assert len(result["protocol_sha256"]) == 64
    assert result["phase"] == "developer_held_out_dataset_types_not_independent_blind_study"
    assert all(row["estimator"] == row["protocol_estimator"] for row in result["records"])
    families = {row["dataset"]: row["estimator"] for row in result["records"]}
    assert families == {
        "wine": "DecisionTreeClassifier",
        "digits": "DecisionTreeClassifier",
        "linnerud": "Ridge",
    }

    by_case = result["by_case"]
    assert by_case["clean"]["label_witness_ok"] == 3
    assert by_case["clean"]["label_witness_problem"] == 0
    assert by_case["permuted_training_targets"]["label_witness_problem"] == 3
    assert by_case["single_modified_target"]["label_witness_problem"] == 3
    assert by_case["no_target_witness"]["label_witness_unknown"] == 3
    assert by_case["no_target_reference"]["label_witness_unknown"] == 3
    assert by_case["corrupted_reference_targets"]["label_witness_problem"] == 3

    assert by_case["permuted_training_targets"]["metadata_alerts"] == 0
    assert by_case["permuted_training_targets"]["feature_content_alerts"] == 0
    assert by_case["permuted_training_targets"]["histogram_or_mean_alerts"] == 0
    assert by_case["permuted_training_targets"]["privileged_direct_comparator_alerts"] == 3
    assert by_case["single_modified_target"]["privileged_direct_comparator_alerts"] == 3
    assert by_case["corrupted_reference_targets"]["privileged_direct_comparator_alerts"] == 3

    # False claims to avoid: the HMAC check cannot determine which side of
    # a witnessed/reference mismatch contains correct labels, and a
    # privileged direct comparator is equally capable of detecting it.
    assert all(row["observed_status"] == "unknown" for row in result["records"] if row["fault"] == "no_target_reference")
