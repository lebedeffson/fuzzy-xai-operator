from __future__ import annotations

from fuzzyxai.category import ExplanationCategory, RiskContext, has_auto_accept
from tests.test_expl_category_laws import _e


def test_characteristic_morphism_auto_accept_smoke() -> None:
    cat = ExplanationCategory(gamma_max=1.0)
    a = cat.object('A', _e('A'))
    b = cat.object('B', _e('B'))
    risk = RiskContext(cat, {a: {'accept', 'block'}, b: {'defer_to_human'}})
    assert has_auto_accept(risk, a)
    assert not has_auto_accept(risk, b)
