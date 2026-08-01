from __future__ import annotations

import importlib.util

import pytest
from fuzzyxai.ml_vertical.service import MLVerticalService
from fuzzyxai.ml_vertical.tracking import ARTIFACTS, log_run


@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="mlflow extra not installed")
def test_real_file_backed_mlflow(tmp_path) -> None:
    run = MLVerticalService().execute(MLVerticalService().scenario_request("S1_NORMAL"))
    result = log_run(run, tracking_uri=tmp_path.as_uri())
    assert result["status"] == "MLFLOW_INTEGRATION_PASS"
    assert tuple(result["artifacts"]) == tuple(ARTIFACTS)
    assert result["run_id"]
