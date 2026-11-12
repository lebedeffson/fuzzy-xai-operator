"""Predeclared four-model fit-provenance developmental experiment regression."""

from __future__ import annotations

from experiments.ml_fit_source_witness_benchmark import run_benchmark


def test_witness_comparison_is_reproducible_and_honest_about_strong_metadata() -> None:
    report = run_benchmark()
    assert report["datasets"] == 4
    assert report["cases"] == 20
    assert report["phase"] == "developer_constructed_developmental_not_confirmatory"
    grouped = report["by_scenario"]

    assert grouped["clean"]["metadata_detections"] == 0
    assert grouped["clean"]["witness_detections"] == 0
    assert grouped["clean"]["unknown_witness"] == 0

    assert grouped["fit_evaluation_truthful_ids"]["metadata_detections"] == 4
    assert grouped["fit_evaluation_truthful_ids"]["witness_detections"] == 4

    assert grouped["fit_evaluation_false_ids"]["metadata_detections"] == 0
    assert grouped["fit_evaluation_false_ids"]["witness_detections"] == 4
    assert grouped["fit_evaluation_false_ids"]["source_content_contradictions"] == 4

    assert grouped["fit_evaluation_true_trace_false_metadata"]["metadata_detections"] == 0
    assert grouped["fit_evaluation_true_trace_false_metadata"]["witness_detections"] == 4

    # No witness does not mean a proven safe fit.
    assert grouped["without_witness"]["metadata_detections"] == 0
    assert grouped["without_witness"]["witness_detections"] == 0
    assert grouped["without_witness"]["unknown_witness"] == 4

    # Training was intentionally contaminated in some scenarios; no claims
    # about generalization performance of those estimators may be made.
    assert "independent" not in report["phase"]
