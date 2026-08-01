from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("shap")
pytest.importorskip("torch")

from fuzzyxai.experiments.h9_e2e_latency import _measure, _pipelines


def test_all_h9_pipelines_measure_four_components() -> None:
    for pipeline in _pipelines():
        row = _measure(pipeline, 1, True)
        assert {"model_ms", "explainer_ms", "fuzzyxai_ms", "serialization_ms", "total_ms"} <= row.keys()
        assert all(np.isfinite(value) and value >= 0 for value in row.values())
