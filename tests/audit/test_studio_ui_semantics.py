from pathlib import Path

from fuzzyxai.studio.operator_scenarios import load_scenarios


def test_studio_source_keeps_raw_trace_behind_technical_trace() -> None:
    source = Path("apps/fuzzyxai_studio.py").read_text(encoding="utf-8")

    assert "Экосистема" in source
    assert "БЛОКИРОВКА" in source
    assert "Технический след" in source
    assert source.index("Технический след") < source.index("ui.code(")
    scenario = next(s for s in load_scenarios() if s["scenario_id"] == "hybrid_xiris")
    assert scenario["diagnostics_summary"][0]["diagnostic_id"] == "D_quality_source_conflict"
