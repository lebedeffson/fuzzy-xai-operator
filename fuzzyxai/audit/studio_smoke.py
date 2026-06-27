from __future__ import annotations

from pathlib import Path

from fuzzyxai.core.scenario_engine import compute_hybrid_xiris
from fuzzyxai.studio.operator_scenarios import build_report, load_scenarios


def main() -> None:
    source = Path("apps/fuzzyxai_studio.py").read_text(encoding="utf-8")
    scenario = next(s for s in load_scenarios() if s["scenario_id"] == "hybrid_xiris")
    report = build_report(scenario)
    result = compute_hybrid_xiris()
    required_source = ["Экосистема", "БЛОКИРОВКА", "Технический след", "Риск-наблюдатель"]
    missing = [item for item in required_source if item not in source]
    if missing:
        raise SystemExit(f"studio source labels missing: {missing}")
    if "ρ = 0.74" in source or "gamma_ij = 0.35" in source:
        raise SystemExit("stale HYBRID value found in Studio source")
    assert result.gamma == 0.351
    assert result.delta == 0.106811
    assert result.rho == 0.8
    assert result.action == "block"
    diagnostic = report["diagnostics"][0]
    assert diagnostic["diagnostic_id"] == "D_quality_source_conflict"
    assert diagnostic["legacy_id"] == "D_source_conflict"
    print("studio-smoke: PASS")


if __name__ == "__main__":
    main()
