from __future__ import annotations

from itertools import combinations

from fuzzyxai.category import ExplanationCategory, RiskContext, auto_accept_subpresheaf
from tests.test_expl_category_laws import _e


def _all_subsets(items):
    items = list(items)
    for r in range(len(items) + 1):
        for comb in combinations(items, r):
            yield set(comb)


def _enumerate_sieves_identity_only(cat: ExplanationCategory, target):
    # Finite demo: for target E we enumerate sieves over Hom(E, E)={id_E}.
    incoming = set(cat.hom(target, target))
    return list(_all_subsets(incoming))


def test_finite_sieves_and_auto_subobject() -> None:
    cat = ExplanationCategory(gamma_max=1.0)
    a = cat.object('A', _e('A'))
    b = cat.object('B', _e('B'))
    c = cat.object('C', _e('C'))
    cat.make_morphism(a, b, name='f', gamma=0.1)
    cat.make_morphism(b, c, name='g', gamma=0.1)

    sieves_c = _enumerate_sieves_identity_only(cat, c)
    assert len(sieves_c) >= 2

    risk = RiskContext(cat, {a: {'accept', 'block'}, b: {'defer_to_human'}, c: {'request_more_data'}})
    auto = auto_accept_subpresheaf(risk)
    assert auto.is_subobject_of(risk)
