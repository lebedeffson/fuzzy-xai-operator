from __future__ import annotations

from fuzzyxai.category import ContextPresheaf, ExplanationCategory, RepresentablePresheaf, yoneda_element_count
from tests.test_expl_category_laws import _e


def test_representable_and_yoneda_count_small_category() -> None:
    cat = ExplanationCategory(gamma_max=1.0)
    a = cat.object('A', _e('A'))
    b = cat.object('B', _e('B'))
    cat.make_morphism(a, b, name='f', gamma=0.1)

    yb = RepresentablePresheaf(cat, b)
    assert len(yb(a)) >= 1
    assert len(yb(b)) >= 1

    p = ContextPresheaf()
    p.add_contexts(a, {'ctx1'})
    p.add_contexts(b, {'ctx2', 'ctx3'})

    assert yoneda_element_count(p, a) == 1
    assert yoneda_element_count(p, b) == 2
