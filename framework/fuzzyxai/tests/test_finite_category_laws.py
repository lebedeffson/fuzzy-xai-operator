from __future__ import annotations

from tests.test_expl_category_laws import (
    test_category_composition_is_associative_on_valid_chain,
    test_category_identity_laws,
)


def test_finite_category_identity_alias() -> None:
    test_category_identity_laws()


def test_finite_category_associativity_alias() -> None:
    test_category_composition_is_associative_on_valid_chain()
