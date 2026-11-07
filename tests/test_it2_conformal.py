"""Split conformal is independent of Type-2 FOU uncertainty."""
from __future__ import annotations

import copy
import math

import pytest

from fuzzyxai import ExplainPlan, FuzzyXAI, IT2SplitConformal
from fuzzyxai.interval_type2 import content_sha256
from tests.test_it2_runtime_adapter import model


def calibrator(n: int = 20, alpha: float = 0.1) -> IT2SplitConformal:
    native = model()
    return IT2SplitConformal.fit_it2(
        native, [[0.0]] * n, [0] * n,
        training_ids=["trainA", "trainB"],
        calibration_ids=[f"cal-{i}" for i in range(n)],
        calibration_dataset_version="control-validation-v1", alpha=alpha,
    )


def test_finite_sample_correction_and_insufficient_n_restricts_not_overstates():
    baseline = calibrator(20, 0.1)
    assert baseline.threshold == pytest.approx(sorted(baseline.calibration_scores)[18])
    # For n=3 and alpha=0.1, ceil(4*0.9)=4>3: no finite threshold.
    tiny = calibrator(3, 0.1)
    assert tiny.threshold is None
    assert tiny.predict_set(0.4, object_id="future")["label_set"] == [0, 1]


def test_training_calibration_and_test_overlap_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        IT2SplitConformal.fit_it2(
            model(), [[0.0], [0.0]], [0, 0],
            training_ids=["cal-1"], calibration_ids=["cal-0", "cal-1"],
            calibration_dataset_version="validation-v1",
        )
    cal = calibrator()
    with pytest.raises(ValueError, match="overlap"):
        cal.predict_set(0.2, object_id="cal-0")
    with pytest.raises(ValueError, match="overlap"):
        cal.predict_set(0.2, object_id="trainA")


def test_invalid_labels_quantiles_and_nonfinite_values_fail_closed():
    native = model()
    digest = content_sha256(native.to_dict())
    with pytest.raises(ValueError, match="binary"):
        IT2SplitConformal.fit_probabilities(
            [0.2], [2], model_sha256=digest,
            calibration_ids=["a"], training_ids=["b"],
            calibration_dataset_version="v1",
        )
    with pytest.raises(ValueError):
        IT2SplitConformal.fit_probabilities(
            [math.nan], [0], model_sha256=digest,
            calibration_ids=["a"], training_ids=["b"],
            calibration_dataset_version="v1",
        )
    cal = calibrator()
    manifest = cal.to_dict()
    manifest["threshold"] = 0.6
    with pytest.raises(ValueError):
        IT2SplitConformal.from_dict(manifest)
    tampered = cal.to_dict()
    tampered["calibration_scores"][0] = 0.5
    with pytest.raises(ValueError):
        IT2SplitConformal.from_dict(tampered)


def test_conformal_can_veto_an_it2_accepted_case():
    native = model()
    cal = calibrator()
    plan = ExplainPlan.default()
    plan.metadata["it2_conformal"] = cal.to_dict()
    ordinary = FuzzyXAI.wrap(native).explain_one(
        [0.2], object_id="unseen", dataset_version="controlled-test-v1",
    )
    assert ordinary.prediction.metadata["it2_guard"]["cases"][0]["decision"]["action"] == "accept"
    classified = FuzzyXAI.wrap(native, explain_plan=plan).explain_one(
        [0.2], object_id="unseen", dataset_version="controlled-test-v1",
    )
    evidence = classified.prediction.metadata["it2_guard"]["conformal"]
    assert evidence["status"] == "verified_calibration_arithmetic"
    assert evidence["cases"][0]["label_set"] == []
    assert evidence["cases"][0]["action"] == "request_more_data"
    assert classified.action != "accept"
    assert classified.explain_it2()["conformal"] == evidence
    assert any(node.node_id == "it2:conformal" for node in classified.explanation_graph.nodes)
    assert any(edge.source == "it2:conformal" and edge.target == "action"
               for edge in classified.explanation_graph.edges)
    assert "input_values" not in repr(classified.explain_it2()["conformal"])


def test_calibrator_rejects_wrong_fitted_model_or_untrusted_anchor():
    cal = calibrator()
    plan = ExplainPlan.default()
    plan.metadata["it2_conformal"] = cal.to_dict()
    with pytest.raises(ValueError, match="different fitted IT2"):
        FuzzyXAI.wrap(model(high_coefficient=3.0), explain_plan=plan).explain_one(
            [0.2], object_id="new", dataset_version="test-v1")
    plan.metadata["it2_conformal_trusted_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="anchor mismatch"):
        FuzzyXAI.wrap(model(), explain_plan=plan).explain_one(
            [0.2], object_id="new", dataset_version="test-v1")


def test_public_calibration_proof_is_hashable_and_replayed():
    cal = calibrator()
    snapshot = cal.to_dict()
    assert IT2SplitConformal.from_dict(snapshot).to_dict() == snapshot
    assert "0.0" not in str(snapshot.get("training_id_hashes"))
    assert snapshot["calibration_dataset_version"] == "control-validation-v1"
    assert len(snapshot["receipt_sha256"]) == 64
