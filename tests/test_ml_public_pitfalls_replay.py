"""Contract of a pre-registered public-documentation replay, NOT a blinded study.

Never turn an unsuccessful scientific replication into a passing positive
performance claim. Detection counts are reported, not tuned to acceptance.
"""

from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache

from experiments.ml_public_pitfalls_replay import run_benchmark


@lru_cache(maxsize=1)
def _report():
    return run_benchmark()


def test_protocol_catalog_and_externally_documented_methods():
    report = _report()
    assert report["phase"] == (
        "public_documentation_scenario_replay_not_blinded_natural_incident_validation"
    )
    assert "scikit-learn.org" in report["source_url"]
    assert report["n_external_examples"] == 3
    assert report["n_cases"] == 9
    assert report["truth_counts"] == {"valid": 3, "invalid": 3, "unverifiable": 3}
    assert len(report["protocol_sha256"]) == 64
    assert Counter(r["example"] for r in report["records"]) == {
        "inconsistent_scaler": 3,
        "feature_selection_leak": 3,
        "pre_split_scaler": 3,
    }
    assert Counter(r["scenario"] for r in report["records"]) == {
        "correct": 3,
        "documented_error": 3,
        "missing_transform_record": 3,
    }


def test_observation_and_comparator_outcomes_are_separately_reported():
    report = _report()
    for method in ("observer", "equal_evidence_direct", "metadata_only"):
        values = report["by_method"][method]
        assert values["true_positive"] + values["false_negative"] == 3
        assert values["false_positive"] + values["true_negative"] == 3
        assert (
            values["abstentions_on_missing"]
            + values["unwarranted_alerts_on_missing"]
            + values["unwarranted_no_alert_on_missing"]
        ) == 3
    for row in report["records"]:
        assert row["observer"] in ("alert", "no_alert", "abstain")
        assert row["metadata_only"] in ("alert", "no_alert", "abstain")
        assert row["equal_evidence_direct"] in ("alert", "no_alert", "abstain")
        if row["truth"] == "invalid":
            assert isinstance(row["hit_designated_rule"], bool)
            assert row["designated_check"] in (
                "ok", "problem", "unknown", "not_applicable",
            )
        else:
            assert row["hit_designated_rule"] is None


def test_unknown_transform_evidence_abstains_not_certifies_clean():
    missing = [r for r in _report()["records"] if r["truth"] == "unverifiable"]
    assert len(missing) == 3
    for record in missing:
        assert record["observer"] == "abstain"
        assert record["equal_evidence_direct"] == "abstain"
        assert record["transform_trace"] == "unknown"
        assert record["preprocessor_fit_rows"] == record["rows_train"]


def test_real_estimators_produce_finite_metrics_but_metrics_do_not_define_faults():
    records = _report()["records"]
    for record in records:
        assert record["rows_train"] > 0 and record["rows_evaluation"] > 0
        assert math.isfinite(record["model_metric_value"])
        assert record["model_metric"] in ("mse", "accuracy")
        assert record["model_metric_value"] >= 0
        if record["model_metric"] == "accuracy":
            assert record["model_metric_value"] <= 1
    assert {r["model_type"] for r in records} == {
        "LinearRegression", "HistGradientBoostingClassifier", "Ridge",
    }
