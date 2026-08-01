from __future__ import annotations

import json
from pathlib import Path

from fuzzyxai.experiments.h10_c5_pilot import (
    _load_lock,
    run_pilot_selection,
)


def test_pilot_lock_binds_parent_results() -> None:
    lock = _load_lock(Path.cwd())
    assert lock["status"] == "LOCKED_BEFORE_SELECTION"
    assert lock["confirmation_claim_permitted"] is False


def test_selection_fails_closed_without_operational_inputs(
    tmp_path: Path,
) -> None:
    # The repository-level integration test covers the actual parent rows.
    result = run_pilot_selection(Path.cwd())
    assert result["status"] == (
        "H10_C5_PILOT_BLOCKED_NO_ELIGIBLE_INCIDENTS"
    )
    assert result["selected_incidents"] == 0
    assert result["gold_accessed_before_execution"] is False
    selection = json.loads(
        Path(
            "results/h10_c5_pilot/"
            "H10_C5_PILOT_SELECTION.json"
        ).read_text(encoding="utf-8")
    )
    assert selection["gold_accessed_during_selection"] is False
    assert tmp_path.exists()
