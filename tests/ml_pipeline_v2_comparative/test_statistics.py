from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ml_pipeline_v2_comparative/run_evaluation.py"
SPEC = importlib.util.spec_from_file_location("comparative_evaluation", SCRIPT)
assert SPEC and SPEC.loader
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_mcnemar_exact_is_deterministic() -> None:
    left = [True, True, True, False]
    right = [False, False, True, False]

    assert evaluation.mcnemar_exact(left, right) == (2, 0, 0.5)


def test_paired_bootstrap_is_reproducible() -> None:
    left = [True, True, False, True]
    right = [False, True, False, False]

    first = evaluation.paired_bootstrap(left, right, seed=1729, iterations=500)
    second = evaluation.paired_bootstrap(left, right, seed=1729, iterations=500)

    assert first == second
    assert first[0] == pytest.approx(0.5)


def test_holm_correction_is_monotonic() -> None:
    rows = [
        {"p_value_raw": 0.01, "p_value_holm": 0.0},
        {"p_value_raw": 0.02, "p_value_holm": 0.0},
        {"p_value_raw": 0.5, "p_value_holm": 0.0},
    ]

    evaluation.holm_adjust(rows)

    assert [row["p_value_holm"] for row in rows] == sorted(row["p_value_holm"] for row in rows)
    assert all(row["p_value_holm"] >= row["p_value_raw"] for row in rows)
