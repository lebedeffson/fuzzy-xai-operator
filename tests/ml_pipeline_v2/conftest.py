from __future__ import annotations

import pytest
from fuzzyxai.ml_vertical.pipeline import MLPipelineService


@pytest.fixture(scope="session")
def pipeline_service() -> MLPipelineService:
    return MLPipelineService()
