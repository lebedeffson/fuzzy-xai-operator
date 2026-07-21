from __future__ import annotations

import numpy as np

from experiments.chapter4_v13.common import canonical_bytes, sha256_bytes, verify_protocol_hash
from experiments.chapter4_v13.run_policies import _actions, _holm
from experiments.chapter4_v13.run_route_faults import clean_route, inject, simple_or, typed_route_validator
from experiments.chapter4_v13.smoke import run


def test_protocol_hash_is_frozen() -> None:
    assert verify_protocol_hash() == "55e86e3bcc2d1f56dfd4700c3912313e411f9d29481109f45c29bf82e77c102e"


def test_canonical_hash_is_order_independent() -> None:
    assert sha256_bytes(canonical_bytes({"a": 1, "b": 2})) == sha256_bytes(canonical_bytes({"b": 2, "a": 1}))


def test_matched_budget_has_exact_review_count() -> None:
    actions = _actions(np.asarray([0.1, 0.4, 0.2, 0.3]), 0.5)
    assert list(actions).count("review") == 2


def test_holm_is_monotone_in_sorted_order() -> None:
    adjusted = _holm([0.01, 0.04, 0.03])
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert adjusted[0] <= adjusted[2] <= adjusted[1]


def test_held_out_fault_distinguishes_typed_validator() -> None:
    route = clean_route(7)
    inject(route, "mixed_model_artifacts")
    assert simple_or(route)[0] == []
    assert typed_route_validator(route)[0] == ["mixed_model_artifacts"]


def test_real_controller_smoke() -> None:
    result = run()
    assert result["trace_id"]
