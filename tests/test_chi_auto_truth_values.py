from __future__ import annotations

from fuzzyxai.category import ExplanationCategory, RiskContext, auto_accept_subpresheaf, has_auto_accept
from tests.test_expl_category_laws import _e


def test_chi_auto_truth_values_match_contexts() -> None:
    cat = ExplanationCategory(gamma_max=1.0)
    e_model = cat.object('E_model', _e('E_model'))
    e_risk = cat.object('E_risk', _e('E_risk'))
    e_action = cat.object('E_action', _e('E_action'))

    risk = RiskContext(
        cat,
        {
            e_model: {'accept', 'lower_confidence', 'request_more_data'},
            e_risk: {'defer_to_human'},
            e_action: {'block'},
        },
    )
    auto = auto_accept_subpresheaf(risk)

    assert has_auto_accept(risk, e_model) is True
    assert has_auto_accept(risk, e_risk) is False
    assert has_auto_accept(risk, e_action) is False

    assert bool(auto(e_model)) is True
    assert bool(auto(e_risk)) is False
    assert bool(auto(e_action)) is False
