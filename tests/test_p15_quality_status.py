"""P15.13/P16 section 19: every quality metric carries a structured record
{value, status, method, reason, source_refs} instead of a bare `None` that
reads like an omission. Metrics that CAN be derived from already-available
evidence (reconstruction_error from model_internals) are now measured
instead of staying permanently `None`.
"""

from __future__ import annotations

from fuzzyxai import FuzzyXAI
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression


def test_reconstruction_error_is_measured_not_permanently_none() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    model = LogisticRegression(max_iter=3000).fit(X, y)
    result = FuzzyXAI.wrap(model).explain_one(X[0], object_id="p0")
    assert result.view_model.quality_metrics["reconstruction_error"] is not None
    status_entry = result.view_model.quality_status["reconstruction_error"]
    assert status_entry["status"] == "measured"
    assert status_entry["reason"] == ""
    assert status_entry["value"] == result.view_model.quality_metrics["reconstruction_error"]
    assert status_entry["method"]
    assert status_entry["source_refs"] == ["model_internals"]


def test_unmeasured_metrics_carry_a_concrete_reason_not_bare_none() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    model = LogisticRegression(max_iter=3000).fit(X, y)
    result = FuzzyXAI.wrap(model).explain_one(X[0], object_id="p0")
    for name, value in result.view_model.quality_metrics.items():
        status_entry = result.view_model.quality_status[name]
        if value is None:
            assert status_entry["status"] == "not_evaluated"
            assert status_entry["reason"]
        else:
            assert status_entry["status"] == "measured"


def test_manually_supplied_metric_is_reported_as_measured() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    model = LogisticRegression(max_iter=3000).fit(X, y)
    result = FuzzyXAI.wrap(model).explain_one(X[0], object_id="p0", evidence={"quality_metrics": {"faithfulness": 0.9}})
    assert result.view_model.quality_metrics["faithfulness"] == 0.9
    status_entry = result.view_model.quality_status["faithfulness"]
    assert status_entry["status"] == "measured"
    assert status_entry["value"] == 0.9


def test_quality_status_survives_json_round_trip(tmp_path) -> None:
    from fuzzyxai.visualization import ExplanationViewModel

    X, y = load_breast_cancer(return_X_y=True)
    model = LogisticRegression(max_iter=3000).fit(X, y)
    result = FuzzyXAI.wrap(model).explain_one(X[0], object_id="p0")
    path = result.export_json(tmp_path / "result.json")
    restored = ExplanationViewModel.load_json(path)
    assert restored.quality_status == result.view_model.quality_status
