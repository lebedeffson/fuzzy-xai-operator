"""Freeze the declared four-dataset single-transform comparison results."""

from __future__ import annotations

from experiments.ml_transform_chain_benchmark import run_benchmark


def test_real_preprocessing_chain_comparison_preserves_all_scenarios() -> None:
    result = run_benchmark()
    assert result["n_pipelines"] == 4
    assert result["n_cases"] == 24
    assert result["phase"] == "developer_constructed_single_preprocessor_not_independent_blind_study"
    assert len(result["protocol_sha256"]) == 64
    assert {row["dataset"]: row["estimator"] for row in result["records"]} == {
        "breast_cancer": "LogisticRegression",
        "wine": "LogisticRegression",
        "diabetes": "Ridge",
        "iris_clustering": "KMeans",
    }
    assert all(row["estimator"] == row["declared_estimator"] for row in result["records"])
    expected = {
        "clean": (4, 0, 0, 0, 0),
        "changed_transformed_model_fit": (4, 0, 4, 4, 0),
        "changed_transformed_inference": (4, 0, 4, 4, 0),
        "changed_declared_raw_evaluation": (4, 0, 4, 4, 0),
        "missing_transform_witness": (4, 0, 0, 0, 4),
        "preprocessor_fit_on_evaluation": (4, 4, 4, 4, 0),
    }
    for case, (n, metadata, direct, problems, unknown) in expected.items():
        actual = result["by_scenario"][case]
        assert (
            actual["n"], actual["metadata_baseline_alerts"],
            actual["direct_comparator_alerts"], actual["transform_problem"],
            actual["transform_missing_unknown"],
        ) == (n, metadata, direct, problems, unknown), (case, actual)
    for record in result["records"]:
        checks = record["checks"]
        case = record["case"]
        if case == "clean":
            assert checks["transform_train_input"] == "ok"
            assert checks["transform_evaluation_input"] == "ok"
            assert checks["transform_preprocessor_fit_link"] == "ok"
            assert checks["transform_model_fit_link"] == "ok"
            assert checks["transform_predict_link"] == "ok"
            if record["task"] == "classification":
                assert checks["transform_predict_proba_link"] == "ok"
            else:
                assert checks["transform_predict_proba_link"] == "not_applicable"
        elif case == "changed_transformed_model_fit":
            assert checks["transform_model_fit_link"] == "problem"
        elif case == "changed_transformed_inference":
            assert checks["transform_predict_link"] == "problem"
        elif case == "changed_declared_raw_evaluation":
            assert checks["transform_evaluation_input"] == "problem"
        elif case == "missing_transform_witness":
            assert checks["transform_trace"] == "unknown"
        elif case == "preprocessor_fit_on_evaluation":
            assert checks["fit_trace_preprocessor_isolation"] == "problem"
            assert checks["transform_preprocessor_fit_link"] == "problem"
