from fuzzyxai.core.diagnostics import DiagnosticType
from fuzzyxai.studio.operator_scenarios import load_scenarios


def test_gd_anfis_shap_operator_example_is_warning_not_block() -> None:
    scenario = next(s for s in load_scenarios() if s["scenario_id"] == "gd_anfis_shap")
    assert scenario["runs"][0]["final_action"] != "block"
    assert scenario["expected_result"] == {}
    assert DiagnosticType.RULE_ATTRIBUTION_CONFLICT.value == "D_rule_attribution_conflict"
