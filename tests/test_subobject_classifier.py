from __future__ import annotations

from fuzzyxai.category import ExplanationCategory, RiskContext, auto_accept_subpresheaf
from tests.test_expl_category_laws import _e


def test_auto_accept_subobject_smoke() -> None:
    cat = ExplanationCategory(gamma_max=1.0)
    a = cat.object('A', _e('A'))
    b = cat.object('B', _e('B'))
    risk = RiskContext(cat, {a: {'accept', 'block'}, b: {'defer_to_human'}})
    auto = auto_accept_subpresheaf(risk)
    assert auto.is_subobject_of(risk)
