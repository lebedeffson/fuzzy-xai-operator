from __future__ import annotations

from fuzzyxai.category import ExplanationCategory
from fuzzyxai.hott import build_temporal_drift_path

from tests.test_expl_category_laws import _e


def test_temporal_drift_path_is_continuous_when_neighbors_align():
    cat = ExplanationCategory({'repr': 0.0, 'uncertainty': 0.0, 'trace': 0.0}, gamma_max=0.5)
    objects = [cat.object(f'E{i}', _e(f'E{i}', 0.2)) for i in range(3)]

    drift = build_temporal_drift_path(cat, ['t0', 't1', 't2'], objects)

    assert drift.is_continuous
    assert len(drift.paths) == 2
    assert not drift.ruptures


def test_temporal_drift_path_records_rupture_when_alignment_breaks():
    cat = ExplanationCategory({'uncertainty': 1.0}, gamma_max=0.2)
    objects = [
        cat.object('E0', _e('E0', 0.1)),
        cat.object('E1', _e('E1', 0.9)),
    ]

    drift = build_temporal_drift_path(cat, ['t0', 't1'], objects)

    assert not drift.is_continuous
    assert len(drift.ruptures) == 1
    assert drift.ruptures[0].diagnostic_state.code == 'D_ij'
