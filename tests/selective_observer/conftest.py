from __future__ import annotations

import pytest

from fuzzyxai.selective_observer import ConfirmatoryProtocolLock


@pytest.fixture
def protocol_lock() -> ConfirmatoryProtocolLock:
    return ConfirmatoryProtocolLock(
        schema_version="1.0",
        frozen_predecessor_commit="1" * 40,
        implementation_commit="2" * 40,
        interface_sha256="a" * 64,
        dictionary_sha256="b" * 64,
        formative_dataset_hashes=("c" * 64,),
        confirmatory_dataset_hashes=("d" * 64,),
        formative_participant_hashes=(),
        confirmatory_participant_hashes=(),
        primary_outcomes=("wrong automatic decisions at matched coverage",),
        preregistered_baselines=("calibrated confidence threshold",),
        minimum_effects=("relative risk reduction >= 0.15",),
        statistical_tests=("paired bootstrap",),
        exclusion_rules=("invalid source provenance",),
        stopping_rule="fixed sample",
        required_sample_size=100,
        independent_timestamp="2026-08-01T00:00:00Z",
        protocol_sha256="e" * 64,
    )
