from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

pytest.importorskip("mlflow")

from fuzzyxai.ml_vertical.pipeline import MLPipelineService
from fuzzyxai.ml_vertical.tracking import PIPELINE_ARTIFACTS, log_pipeline_run


def test_real_mlflow_run_contains_complete_pipeline_artifacts(tmp_path: Path) -> None:
    run = MLPipelineService().execute_scenario("S16_MODEL_ARTIFACT_TAMPER")
    result = log_pipeline_run(run, tracking_uri=(tmp_path / "mlruns").as_uri())
    artifact_root = Path(unquote(urlparse(result["artifact_uri"]).path)) / "pipeline"

    assert result["status"] == "PIPELINE_MLFLOW_INTEGRATION_PASS"
    assert set(result["artifacts"]) == set(PIPELINE_ARTIFACTS)
    assert all((artifact_root / name).is_file() for name in PIPELINE_ARTIFACTS)
