from __future__ import annotations

from fuzzyxai.category import ExplanationCategory, try_make_morphism
from fuzzyxai.hott import ExplanationPath, RuptureType

from tests.test_expl_category_laws import _e


def test_explanation_paths_concat_and_validate():
    cat = ExplanationCategory(gamma_max=1.0)
    a = cat.object('A', _e('A'))
    b = cat.object('B', _e('B'))
    c = cat.object('C', _e('C'))
    f = cat.make_morphism(a, b, gamma=0.2)
    g = cat.make_morphism(b, c, gamma=0.3)

    path = ExplanationPath.from_morphism(f).concat(ExplanationPath.from_morphism(g))

    assert path.source == 'A'
    assert path.target == 'C'
    assert path.gamma_total == 0.5
    assert path.is_valid(gamma_max=0.5)


def test_rupture_type_wraps_failed_path():
    cat = ExplanationCategory(gamma_max=0.1)
    a = cat.object('A', _e('A'))
    b = cat.object('B', _e('B'))
    result = try_make_morphism(cat, a, b, gamma=0.9)

    rupture = RuptureType.from_diagnostic('A', 'B', result.diagnostic, gamma=0.9, threshold=0.1)

    assert result.success is False
    assert rupture.source == 'A'
    assert rupture.target == 'B'
    assert rupture.diagnostic_state.code == 'D_ij'
