from __future__ import annotations

import pytest
from fuzzyxai.pipelines.practical import CrossPipelineService


@pytest.fixture(scope="session")
def service() -> CrossPipelineService:
    instance = CrossPipelineService()
    instance.prepare_all()
    return instance
