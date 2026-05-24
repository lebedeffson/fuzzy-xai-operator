from __future__ import annotations

from fuzzyxai.category import ContextPresheaf, ExplanationCategory, RepresentablePresheaf, yoneda_element_count

from tests.test_expl_category_laws import _e


def test_representable_presheaf_lists_hom_to_target():
    cat = ExplanationCategory(gamma_max=1.0)
    x = cat.object('X', _e('X'))
    y = cat.object('Y', _e('Y'))
    f = cat.make_morphism(x, y, name='f', gamma=0.1)

    yy = RepresentablePresheaf(cat, y)

    assert f in yy(x)
    assert cat.identity(y) in yy(y)


def test_yoneda_demo_count_matches_context_fiber_size():
    cat = ExplanationCategory(gamma_max=1.0)
    x = cat.object('X', _e('X'))
    p = ContextPresheaf()
    p.add_contexts(x, {'ctx1', 'ctx2'})

    assert yoneda_element_count(p, x) == 2
