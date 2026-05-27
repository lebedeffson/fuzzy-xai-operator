from __future__ import annotations

from fuzzyxai.category import ExplanationCategory, RiskContext
from tests.test_expl_category_laws import _e


def test_context_restrictions_are_functorial() -> None:
    cat = ExplanationCategory(gamma_max=1.0)
    a = cat.object('A', _e('A'))
    b = cat.object('B', _e('B'))
    c = cat.object('C', _e('C'))
    f = cat.make_morphism(a, b, name='f', gamma=0.1)
    g = cat.make_morphism(b, c, name='g', gamma=0.1)
    gf = cat.compose(f, g)

    risk = RiskContext(cat, {a: {'accept'}, b: {'lower_confidence'}, c: {'defer_to_human'}})
    risk.set_restriction(f, lambda action: 'accept')
    risk.set_restriction(g, lambda action: 'lower_confidence')
    risk.set_restriction(gf, lambda action: 'accept')

    assert risk.check_identity(a, cat.identity(a))
    assert risk.check_contravariant_composition(f, g, gf)
