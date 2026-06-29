from __future__ import annotations

from fuzzyxai.category import ExplanationCategory, try_make_morphism

from tests.test_expl_category_laws import _e


def test_diagnostic_completion_returns_morphism_or_diagnostic():
    cat = ExplanationCategory(gamma_max=0.5)
    a = cat.object('A', _e('A'))
    b = cat.object('B', _e('B'))

    ok = try_make_morphism(cat, a, b, gamma=0.2)
    bad = try_make_morphism(cat, a, b, gamma=0.8)

    assert ok.success is True
    assert ok.morphism is not None
    assert bad.success is False
    assert bad.diagnostic is not None
    assert bad.diagnostic.code == 'D_ij'
