"""Real-pipeline developmental comparison of read-only ML observation.

This benchmark creates both genuine fit contamination and intentionally
corrupted downstream records. It is not independent blind validation.
"""

from __future__ import annotations

from experiments.ml_lifecycle_observer_benchmark import run_benchmark


def test_observer_detects_declared_pipeline_faults_without_overclaiming() -> None:
    report = run_benchmark()
    assert report["n_pipelines"] == 4
    assert report["n_cases"] >= 30
    assert report["phase"] == "developmental_known_faults_not_external_validation"

    for case, expected, status in (
        ("preprocessor_fit_includes_evaluation", 1, "problem"),
        ("model_fit_includes_evaluation", 3, "problem"),
        ("split_row_id_overlap", 4, "problem"),
        ("feature_columns_permuted", 4, "problem"),
        ("copied_numeric_row_new_id", 4, "warning"),
        ("numeric_mean_shift", 4, "warning"),
        ("invalid_probability_rows", 2, "problem"),
        ("missing_source_identifiers", 4, "unknown"),
    ):
        group = report["by_case"][case]
        assert group["n_pipelines_applicable"] == expected, (case, group)
        assert all(outcome["status"] == status for outcome in group["target_outcomes"]), (case, group)

    # Strong ordinary baselines already catch straightforward ID/schema
    # problems. The observer must not claim novelty for those checks.
    assert report["by_case"]["model_fit_includes_evaluation"]["metadata_baseline_alerts"] == 3
    assert report["by_case"]["preprocessor_fit_includes_evaluation"]["metadata_baseline_alerts"] == 1
    assert report["by_case"]["feature_columns_permuted"]["metadata_baseline_alerts"] == 4
    assert report["by_case"]["invalid_probability_rows"]["plain_baseline_alerts"] == 2

    # Content collisions with new IDs and descriptive mean shift are
    # additional alerts, but they are *not proof* of defective data.
    assert report["by_case"]["copied_numeric_row_new_id"]["metadata_baseline_alerts"] == 0
    assert report["by_case"]["numeric_mean_shift"]["metadata_baseline_alerts"] == 0

    # Shuffled external ground-truth labels can still look structurally
    # consistent. Neither observer nor baseline validates their semantics.
    for row in report["cases"]:
        if row["case"] == "evaluation_labels_permuted":
            assert row["observer_metrics"]
            assert not any(check["code"] == "label_metrics" and check["status"] == "problem" for check in row["observer_flags"])
