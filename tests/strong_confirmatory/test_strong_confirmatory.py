from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from fuzzyxai.strong_confirmatory import (
    FAULT_TYPES,
    compare_grid_configurations,
    evaluate_route_guardrails,
    holm_adjust,
    paired_bootstrap_difference,
    run_streaming_scalability,
)


ROOT = Path(__file__).resolve().parents[2]


def test_paired_statistics_report_effect_interval_n_and_holm() -> None:
    report = paired_bootstrap_difference([0.8, 0.9, 0.7], [0.6, 0.7, 0.5], repetitions=200, seed=7)
    assert report["n"] == 3
    assert np.isclose(report["effect"], 0.2)
    assert len(report["confidence_interval_95"]) == 2
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert adjusted[0] <= adjusted[1]


def test_route_validity_is_separate_from_model_error_prediction() -> None:
    records = []
    for fault in (None, *FAULT_TYPES):
        for index in range(20):
            records.append(
                {
                    "fault_type": fault,
                    "fault_source": None if fault is None else f"source:{fault}",
                    "detected_fault_type": fault,
                    "detected_fault_source": None if fault is None else f"source:{fault}",
                    "confidence": 0.95,
                    "data_quality": 0.95,
                    "provenance_present": fault != "missing_provenance",
                    "schema_valid": True,
                    "generic_risk": 0.1,
                    "object_id": f"{fault}:{index}",
                }
            )
    report = evaluate_route_guardrails(records)
    typed = next(row for row in report["methods"] if row["method"] == "typed_route_validity")
    assert typed["f1"] == 1.0
    assert report["model_error_prediction_claim_allowed"] is False
    assert report["confirmatory_claim_allowed"] is False


def test_grid_sensitivity_keeps_formative_boundary() -> None:
    base = {
        "actions": ["accept", "review"],
        "representations": ["F0", "Fint"],
        "risk": [0.1, 0.5],
        "top_k": [[1, 2], [2, 3]],
    }
    report = compare_grid_configurations({"default": base, "fine": dict(base)})
    assert report["formative_target_met"] is True
    assert report["confirmatory_claim_allowed"] is False


def test_scalability_smoke_is_deterministic_and_excludes_explainer() -> None:
    report = run_streaming_scalability(sizes=(1_000, 2_000), batch_size=500, seed=9)
    assert report["deterministic_repeat"] is True
    assert report["local_explainer_included"] is False
    assert report["confirmatory_claim_allowed"] is False


def test_protocol_preserves_negative_results_and_lock_fails_closed() -> None:
    subprocess.run([sys.executable, "scripts/strong_confirmatory/build_protocol.py"], cwd=ROOT, check=True)
    frozen = json.loads((ROOT / "study/strong_confirmatory/frozen_claims.json").read_text(encoding="utf-8"))
    assert frozen["immutable_original_statuses"] == {
        "H3-original": "not_supported",
        "H5-P-original": "not_supported",
        "H6-general": "not_supported",
    }
    lock = subprocess.run(
        [sys.executable, "scripts/strong_confirmatory/lock_protocol.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert lock.returncode == 2
    assert "BLOCKED: strong_confirmatory_protocol_lock" in lock.stdout
    assert not (ROOT / "study/strong_confirmatory/confirmatory_protocol_lock.json").exists()
