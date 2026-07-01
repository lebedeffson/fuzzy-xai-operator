from __future__ import annotations

from fuzzyxai.adapters.rule_attribution import adapt_gd_anfis_shap


def get_input():
    return adapt_gd_anfis_shap(
        {
            "x1": 0.88,
            "x2": 0.22,
            "x1_term": "high",
            "x2_term": "low",
            "alpha_rule": 0.82,
            "shap_x1": 0.45,
            "shap_x2": -0.30,
            "gamma_rule_shap": 0.685,
        }
    )
