from fuzzyxai.studio.operator_scenarios import load_scenarios


def test_scenarios_have_honest_evidence_levels() -> None:
    scenarios = {scenario["scenario_id"]: scenario for scenario in load_scenarios()}

    assert scenarios["hybrid_xiris"]["evidence_level"] == "full_control_run"
    assert scenarios["gd_anfis_shap"]["evidence_level"] == "operator_control_example"
    assert scenarios["gis_integro"]["evidence_level"] == "operator_control_example"
    assert scenarios["beacon_xai"]["evidence_level"] in {"route_demonstration", "operator_control_example"}
    for scenario_id, scenario in scenarios.items():
        if scenario_id != "hybrid_xiris":
            assert scenario["evidence_level"] != "full_control_run"
