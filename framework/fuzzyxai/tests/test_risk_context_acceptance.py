from __future__ import annotations

from fuzzyxai.category import ExplanationCategory
from fuzzyxai.risk.context_acceptance import observer_auto_accept_allowed

from tests.test_expl_category_laws import _e


def test_observer_auto_accept_context_gate():
    cat = ExplanationCategory(gamma_max=1.0)
    obj = cat.object('E_model', _e('model'))

    assert observer_auto_accept_allowed(cat, obj, {'accept'})
    assert not observer_auto_accept_allowed(cat, obj, {'block', 'defer_to_human'})
