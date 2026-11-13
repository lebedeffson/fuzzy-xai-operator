"""Regression contract for a frozen multi-transformer transfer benchmark.

The eight pipelines are deliberately different from the StandardScaler-only
development trial. They still share datasets and developer-controlled faults.
"""

from __future__ import annotations

from collections import Counter

from experiments.ml_transform_family_transfer_benchmark import run_benchmark


def test_transfer_declared_pipelines_and_fault_classes() -> None:
    study = run_benchmark()
    assert study["n_pipelines"] == 8
    assert study["n_cases"] == 64
    assert study["n_unique_source_datasets"] == 6
    assert len(study["protocol_sha256"]) == 64
    assert study["phase"] == (
        "developer_specified_cross_transform_family_transfer_not_independent_blind_incident_study"
    )
    assert Counter(r["pipeline"] for r in study["records"]) == {
        name: 8 for name in {
            "digits_pca_knn", "wine_robust_svc",
            "breast_cancer_quantile_logreg", "diabetes_spline_ridge",
            "linnerud_minmax_ridge", "iris_power_rf",
            "digits_maxabs_gaussiannb", "iris_standard_kmeans",
        }
    }
    totals = study["by_scenario"]
    for name in ("clean", "valid_shifted_evaluation", "valid_retrained"):
        assert totals[name] == {
            "n": 8, "metadata_alerts": 0, "direct_alerts": 0,
            "observer_alerts": 0, "observer_unknowns": 0,
        }, (name, totals[name])
    for name in (
        "changed_model_fit_values", "changed_inference_values",
        "changed_reported_raw_evaluation",
    ):
        assert totals[name] == {
            "n": 8, "metadata_alerts": 0, "direct_alerts": 8,
            "observer_alerts": 8, "observer_unknowns": 0,
        }, (name, totals[name])
    assert totals["preprocessor_fit_on_evaluation"] == {
        "n": 8, "metadata_alerts": 8, "direct_alerts": 8,
        "observer_alerts": 8, "observer_unknowns": 0,
    }
    assert totals["missing_transform_witness"] == {
        "n": 8, "metadata_alerts": 0, "direct_alerts": 0,
        "observer_alerts": 0, "observer_unknowns": 8,
    }


def test_each_corruption_detected_at_its_declared_link_not_by_metric_tuning() -> None:
    study = run_benchmark()
    for row in study["records"]:
        if row["truth"] == "invalid":
            assert row["observer_alert"]
            assert row["equal_evidence_direct_alert"]
            assert row["designated_status"] == "problem", row
        elif row["truth"] == "valid":
            assert not row["observer_alert"], row
            assert not row["metadata_alert"], row
            assert not row["equal_evidence_direct_alert"], row
            assert row["checks"]["transform_train_input"] == "ok"
            assert row["checks"]["transform_evaluation_input"] == "ok"
            assert row["checks"]["transform_model_fit_link"] == "ok"
            assert row["checks"]["transform_predict_link"] == "ok"
        else:
            assert row["truth"] == "unverifiable"
            assert row["transform_trace"] == "unknown"
            assert row["observer_missing"]
            assert not row["observer_alert"]


def test_feature_width_change_is_real_not_just_a_declared_tag() -> None:
    study = run_benchmark()
    clean = {r["pipeline"]: r for r in study["records"] if r["scenario"] == "clean"}
    assert clean["digits_pca_knn"]["transformed_feature_count"] == 12
    assert clean["digits_pca_knn"]["raw_feature_count"] == 64
    assert clean["diabetes_spline_ridge"]["transformed_feature_count"] > clean["diabetes_spline_ridge"]["raw_feature_count"]
    assert len({r["transformer"] for r in clean.values()}) == 8
