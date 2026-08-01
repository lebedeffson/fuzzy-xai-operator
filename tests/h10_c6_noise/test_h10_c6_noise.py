from __future__ import annotations

import json
from pathlib import Path

import pytest
from fuzzyxai.experiments.h10_c6_noise import _jaccard, _validate_lock


def test_nonempty_cut_jaccard() -> None:
    assert _jaccard(("a", "b"), ("b", "c")) == pytest.approx(1 / 3)
    assert _jaccard(("a",), ()) == 0.0


def test_empty_baseline_cut_is_not_stability_evidence() -> None:
    with pytest.raises(ValueError, match="empty baseline"):
        _jaccard((), ())


def test_lock_validation_fails_closed_on_changed_dataset(tmp_path: Path) -> None:
    results = tmp_path / "results/h10_c6_noise"
    data = tmp_path / "data/h10_c6_noise"
    results.mkdir(parents=True)
    data.mkdir(parents=True)
    (data / "bank_marketing_uci_222.zip").write_bytes(b"changed")
    objects = results / "H10_C6_N_OBJECT_IDS.json"
    objects.write_text(json.dumps({"object_count": 1000, "objects": []}), encoding="utf-8")
    (results / "H10_C6_N_PROTOCOL_LOCK.json").write_text(
        json.dumps({"status": "LOCKED_BEFORE_EXECUTION", "dataset_sha256": "0" * 64, "object_ids_sha256": "0" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dataset hash mismatch"):
        _validate_lock(tmp_path)
