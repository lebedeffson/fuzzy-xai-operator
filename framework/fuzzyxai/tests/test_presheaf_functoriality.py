from __future__ import annotations

from fuzzyxai.category import ContextPresheaf, ExplanationCategory, PresheafToposDescriptor

from tests.test_expl_category_laws import _e


def test_presheaf_identity_and_contravariant_composition():
    cat = ExplanationCategory(gamma_max=1.0)
    a = cat.object('A', _e('A'))
    b = cat.object('B', _e('B'))
    c = cat.object('C', _e('C'))
    f = cat.make_morphism(a, b, gamma=0.1)
    g = cat.make_morphism(b, c, gamma=0.1)
    gf = cat.compose(f, g)

    presheaf = ContextPresheaf()
    presheaf.add_contexts(a, {'ctx_A'})
    presheaf.add_contexts(b, {'ctx_B'})
    presheaf.add_contexts(c, {'ctx_C'})
    presheaf.set_restriction(cat.identity(a), lambda x: x)
    presheaf.set_restriction(f, lambda x: 'ctx_A' if x == 'ctx_B' else x)
    presheaf.set_restriction(g, lambda x: 'ctx_B' if x == 'ctx_C' else x)
    presheaf.set_restriction(gf, lambda x: 'ctx_A' if x == 'ctx_C' else x)

    assert presheaf.check_identity(a, cat.identity(a))
    assert presheaf.check_contravariant_composition(f, g, gf)
    assert PresheafToposDescriptor().contains(presheaf)
