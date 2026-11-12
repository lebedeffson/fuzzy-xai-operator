"""Predeclared four-model output provenance development benchmark regression."""

from __future__ import annotations

from experiments.ml_output_witness_development_benchmark import run_benchmark


def test_output_witness_benchmark_with_equally_privileged_baseline() -> None:
    result = run_benchmark()
    assert result["n_datasets"] == 4
    assert result["n_cases"] == 22
    assert len(result["protocol_sha256"]) == 64
    assert result["phase"] == "developer_designed_controlled_faults_not_external_blind_validation"
    model_families = {entry["dataset"]: entry["estimator"] for entry in result["records"]}
    assert model_families == {
        "iris": "DecisionTreeClassifier",
        "wine": "RandomForestClassifier",
        "diabetes": "LinearRegression",
        "iris_clustering": "KMeans",
    }
    expected = {
        "clean": (4, 0, 0, 0, 0),
        "reordered_prediction_values": (4, 2, 4, 4, 0),
        "reordered_report_features": (4, 0, 4, 4, 0),
        "reordered_report_ids": (4, 0, 4, 4, 0),
        "missing_output_record": (4, 0, 0, 0, 4),
        "permuted_probability_columns_with_consistent_class_map": (2, 0, 2, 2, 0),
    }
    for case, (n, basic, direct, observed, unknown) in expected.items():
        actual = result["by_scenario"][case]
        assert actual["n"] == n, (case, actual)
        assert actual["basic_numeric_alerts"] == basic, (case, actual)
        assert actual["privileged_direct_alerts"] == direct, (case, actual)
        assert actual["observed_problem"] == observed, (case, actual)
        assert actual["missing_unknown"] == unknown, (case, actual)

    # Ordinary numeric/probability consistency is already a strong baseline:
    # it detects classifier prediction/probability disagreement, so we must
    # not credit the output witness as uniquely discovering those two cases.
    for record in result["records"]:
        checks = record["checks"]
        if record["case"] == "reordered_prediction_values":
            assert checks["output_trace_predict_values"] == "problem"
        if record["case"] == "reordered_report_features":
            assert checks["output_trace_predict_inputs"] == "problem"
        if record["case"] == "reordered_report_ids":
            assert checks["output_trace_predict_row_ids"] == "problem"
        if record["case"] == "permuted_probability_columns_with_consistent_class_map":
            assert checks["output_trace_predict_proba_values"] == "problem"
            assert checks["output_trace_predict_proba_classes"] == "problem"
