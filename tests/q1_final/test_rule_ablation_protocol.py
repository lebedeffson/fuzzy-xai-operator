from __future__ import annotations

import pytest

from fuzzyxai.q1_final.rule_ablation import FROZEN_SUBGROUPS, _holm, _jaccard


def test_confirmatory_subgroups_are_frozen_before_splitting() -> None:
    assert FROZEN_SUBGROUPS == {"uci_covertype": 3, "uci_adult": 1}


def test_path_redundancy_is_a_bounded_jaccard_measure() -> None:
    assert _jaccard((1, 2, 3), (2, 3, 4)) == pytest.approx(0.5)
    assert _jaccard((), ()) == 0.0


def test_holm_correction_preserves_original_order_and_monotonicity() -> None:
    corrected = _holm((0.03, 0.001, 0.02))
    assert corrected == pytest.approx((0.04, 0.003, 0.04))
