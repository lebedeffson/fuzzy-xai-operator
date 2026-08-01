from __future__ import annotations

from dataclasses import asdict

import pytest
from fuzzyxai.ml_vertical.contracts import Evidence
from fuzzyxai.ml_vertical.math import probabilistic_sum, product_tnorm, triangular
from fuzzyxai.ml_vertical.service import SCENARIOS, MLVerticalService

EXPECTED = {
    "S1_NORMAL": ("ACCEPT", "F0"),
    "S2_EXPLAINER_VERSION_MISMATCH": ("BLOCK", "F0"),
    "S3_MISSING_REQUIRED_FEATURE": ("REQUEST_DATA", "F0"),
    "S4_MODEL_RULE_CONFLICT": ("REVIEW", "NAS"),
    "S5_INTERVAL_UNCERTAINTY": ("WARN", "F_int"),
    "S6_MULTILEVEL_UNCERTAINTY": ("REVIEW", "F_ML"),
    "S7_REDUCTION_LOSS_EXCEEDED": ("WARN", "F_int"),
    "S8_INCOMPLETE_PROVENANCE": ("BLOCK", "F0"),
    "S9_REGISTERED_REPAIR": ("ACCEPT", "F0"),
    "S10_DETERMINISM": ("ACCEPT", "F0"),
}


@pytest.fixture(scope="module")
def service() -> MLVerticalService:
    return MLVerticalService()


def test_zero_evidence_is_not_missing() -> None:
    assert Evidence("e:0", "numeric", 0, "fixture").status == "observed"
    with pytest.raises(ValueError):
        Evidence("e:none", "numeric", None, "fixture")


def test_registered_fuzzy_operators() -> None:
    assert triangular(0.5, 0.25, 0.5, 0.75) == 1.0
    assert product_tnorm(0.5, 0.4) == pytest.approx(0.2)
    assert probabilistic_sum(0.5, 0.4) == pytest.approx(0.7)


@pytest.mark.parametrize("scenario_id", tuple(SCENARIOS))
def test_registered_scenario(service: MLVerticalService, scenario_id: str) -> None:
    run = service.execute(service.scenario_request(scenario_id))
    expected_action, expected_representation = EXPECTED[scenario_id]
    assert run.observer["action"] == expected_action
    assert run.representation["representation_id"] == expected_representation
    assert len(run.route_graph["nodes"]) == 13
    assert len(run.route_graph["edges"]) == 12
    assert all(edge["relation_status"] == "known_valid" for edge in run.route_graph["edges"])
    assert {view["explainable_object_sha256"] for view in run.views.values()} == {
        run.views["user"]["explainable_object_sha256"]
    }
    for claim in run.claims:
        assert claim["evidence_refs"]


def test_real_model_and_shap_output_consistency(service: MLVerticalService) -> None:
    run = service.execute(service.scenario_request("S1_NORMAL"))
    assert run.prediction["model_id"] == "bcw-logistic-regression"
    assert run.explanation["explainer_id"] == "shap.LinearExplainer"
    assert run.explanation["output_difference"] <= 1e-8
    assert len(run.explanation["shap_values"]) == 30


def test_critical_defect_cannot_be_compensated(service: MLVerticalService) -> None:
    run = service.execute(service.scenario_request("S2_EXPLAINER_VERSION_MISMATCH"))
    assert run.observer["action"] == "BLOCK"
    assert "MODEL_EXPLAINER_VERSION" in run.observer["critical_issues"]


def test_missing_feature_is_not_imputed(service: MLVerticalService) -> None:
    request = service.scenario_request("S3_MISSING_REQUIRED_FEATURE")
    assert "mean radius" not in request.features
    run = service.execute(request)
    assert run.prediction is None
    assert run.observer["action"] == "REQUEST_DATA"


def test_registered_repair_recertifies_full_route(service: MLVerticalService) -> None:
    run = service.execute(service.scenario_request("S9_REGISTERED_REPAIR"))
    assert run.repair["recertification"]["status"] == "full_success"
    assert run.repair["recertification"]["route_valid_after"] is True
    assert run.repair["recertification"]["new_critical_issues"] == ()


def test_determinism(service: MLVerticalService) -> None:
    request = service.scenario_request("S10_DETERMINISM")
    first = service.execute(request)
    second = service.execute(request)
    assert first.canonical_sha256 == second.canonical_sha256
    assert asdict(first) == asdict(second)


@pytest.mark.parametrize("forbidden", ["target", "gold_patch", "fix_commit", "changed_files"])
def test_gold_channels_fail_closed(service: MLVerticalService, forbidden: str) -> None:
    request = asdict(service.scenario_request("S1_NORMAL"))
    request["controls"][forbidden] = "forbidden"
    with pytest.raises(ValueError, match="forbidden pre-scoring channel"):
        service.execute(request)
