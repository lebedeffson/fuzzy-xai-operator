from __future__ import annotations

from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")

from examples.integrations.mlflow_fuzzyxai_example import run_example


def test_local_mlflow_registry_builds_verified_route(
    tmp_path: Path,
) -> None:
    result = run_example(tmp_path)
    assert result["status"] == "MLFLOW_INTEGRATION_PASS"
    assert result["external_network_required"] is False
    route = (
        tmp_path / "mlflow_fuzzyxai_route.json"
    ).read_text(encoding="utf-8")
    assert result["run_id"] in route
    assert '"node_id": "mlflow_run"' in route
    assert '"node_id": "mlflow_registered_model"' in route
    assert '"edge_id": "edge_mlflow_run_model"' in route
    assert '"edge_id": "edge_mlflow_model_explanation"' in route
