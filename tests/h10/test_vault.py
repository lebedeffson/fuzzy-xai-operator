from __future__ import annotations

import pytest

from experiments.h10.run_confirmatory import _assert_scoring_not_previously_opened
from experiments.h10.vault import open_vault, seal


def test_vault_round_trip_and_integrity() -> None:
    key = b"k" * 32
    payload = b'{"labels":"sealed"}'
    encrypted = seal(payload, key)
    assert payload not in encrypted
    assert open_vault(encrypted, key) == payload
    corrupted = encrypted[:-1] + bytes((encrypted[-1] ^ 1,))
    with pytest.raises(ValueError, match="integrity"):
        open_vault(corrupted, key)


def test_existing_opening_marker_prevents_repeated_scoring(tmp_path) -> None:
    marker = tmp_path / "opening_record.json"
    marker.write_text('{"opening_count":1}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="already been opened"):
        _assert_scoring_not_previously_opened(marker)
