from __future__ import annotations

from fuzzyxai.category import ExplanationCategory
from fuzzyxai.core.explanation_object import ExplanationObject, Rule, Trace
from fuzzyxai.hierarchy.f0 import F0


def _e(name: str, value: float = 0.2) -> ExplanationObject:
    rule = Rule(f'r_{name}', {'risk': 'medium'}, 'review')
    return ExplanationObject(
        terms={'low', 'medium', 'high'},
        representation=F0(lambda x: value, 'risk'),
        rules=[rule],
        activations={rule.name: value},
        uncertainty=value,
        trace=Trace(name, 'v1', 't0', checksum=name),
    )


def test_category_identity_laws():
    cat = ExplanationCategory(gamma_max=1.0)
    a = cat.object('A', _e('A'))
    b = cat.object('B', _e('B', 0.3))
    f = cat.make_morphism(a, b, gamma=0.1)

    left = cat.compose(cat.identity(a), f)
    right = cat.compose(f, cat.identity(b))

    assert left.source == f.source and left.target == f.target
    assert right.source == f.source and right.target == f.target
    assert left.gamma == f.gamma == right.gamma


def test_category_composition_is_associative_on_valid_chain():
    cat = ExplanationCategory(gamma_max=1.0)
    a = cat.object('A', _e('A'))
    b = cat.object('B', _e('B'))
    c = cat.object('C', _e('C'))
    d = cat.object('D', _e('D'))
    f = cat.make_morphism(a, b, gamma=0.1)
    g = cat.make_morphism(b, c, gamma=0.2)
    h = cat.make_morphism(c, d, gamma=0.3)

    left = cat.compose(cat.compose(f, g), h)
    right = cat.compose(f, cat.compose(g, h))

    assert left.source == right.source == a
    assert left.target == right.target == d
    assert round(left.gamma, 10) == round(right.gamma, 10) == 0.6
