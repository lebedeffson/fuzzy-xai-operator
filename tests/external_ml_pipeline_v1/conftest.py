from __future__ import annotations

from pathlib import Path

import pytest

from experiments.external_ml_pipeline_v1.benchmark import ExternalBenchmark


@pytest.fixture(scope="session")
def benchmark() -> ExternalBenchmark:
    root = Path(__file__).resolve().parents[2] / "experiments/external_ml_pipeline_v1/fixtures"
    return ExternalBenchmark(root)
