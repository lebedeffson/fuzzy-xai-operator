from __future__ import annotations

import numpy as np

from apps.chapter5_web_demo import build_backend, evaluate_vector


def test_chapter5_web_demo_backend_smoke():
    backend = build_backend(seed=1)
    vec = backend.x_test.iloc[0].to_numpy(dtype=float)
    out = evaluate_vector(backend, vec, sample_id='test0')
    assert 0.0 <= out['I_pre'] <= 1.0
    assert 0.0 <= out['rho'] <= 1.0
    assert out['action'] in {'accept', 'lower_confidence', 'request_more_data', 'defer_to_human', 'block'}
    assert isinstance(out['contexts'], list) and len(out['contexts']) == 3
