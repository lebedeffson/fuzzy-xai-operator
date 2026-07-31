from __future__ import annotations

from pathlib import Path


def test_existing_layered_ui_displays_pipeline_canonical_result_only() -> None:
    source = (Path(__file__).resolve().parents[2] / "apps/layered_demo.py").read_text(encoding="utf-8")

    assert "_render_ml_pipeline_panel" in source
    assert "FuzzyXAI ML Pipeline v2" in source
    assert "stage_statuses" in source
    assert "violated_contract" in source
    assert "recertification" in source
    assert "pipeline_service.execute_scenario" in source
