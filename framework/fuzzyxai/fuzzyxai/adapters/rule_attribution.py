from __future__ import annotations

from .base import BaseAdapter


class RuleAttributionAdapter(BaseAdapter):
    repo_id = "fims9000/XAI-2.0-SHAP-regularized-ANFIS"
    scenario_id = "gd_anfis_shap"
    required_fields = ("x1", "x2", "x1_term", "x2_term", "alpha_rule", "shap_x1", "shap_x2")


def adapt_gd_anfis_shap(payload: dict) -> object:
    return RuleAttributionAdapter().to_adapted_input(payload)
