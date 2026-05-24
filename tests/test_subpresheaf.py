from __future__ import annotations

from fuzzyxai.category import ExplanationCategory, RiskContext, auto_accept_subpresheaf, has_auto_accept

from tests.test_expl_category_laws import _e


def test_auto_accept_subpresheaf_is_subobject():
    cat = ExplanationCategory(gamma_max=1.0)
    a = cat.object('A', _e('A'))
    b = cat.object('B', _e('B'))
    risk = RiskContext(cat, {a: {'accept', 'block'}, b: {'defer_to_human'}})

    auto = auto_accept_subpresheaf(risk)

    assert auto(a) == {'accept'}
    assert auto(b) == set()
    assert auto.is_subobject_of(risk)
    assert has_auto_accept(risk, a)
    assert not has_auto_accept(risk, b)
