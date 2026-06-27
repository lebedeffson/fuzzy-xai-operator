from fuzzyxai.studio.operator_scenarios import load_scenarios


def test_beacon_batch_and_fragment_levels_are_not_mixed() -> None:
    scenario = next(s for s in load_scenarios() if s["scenario_id"] == "beacon_xai")
    summary = scenario["summary"]

    assert summary["objects_total"] == 100
    assert summary["counter_evidence_detected"] == 83
    assert summary["audit_reports"] == 12
    assert summary["checks_without_beacon"] == 64
    assert summary["checks_with_beacon"] == 11
    assert scenario["evidence_level"] != "full_control_run"
