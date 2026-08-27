from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_base_wheel_has_no_hard_nicegui_dependency() -> None:
    config = tomllib.loads((ROOT / "framework/fuzzyxai/pyproject.toml").read_text(encoding="utf-8"))
    dependencies = config["project"]["dependencies"]
    assert not any(item.lower().startswith("nicegui") for item in dependencies)
    assert any(item.lower().startswith("nicegui") for item in config["project"]["optional-dependencies"]["ui"])


def test_disconnected_research_packages_are_excluded_from_wheel() -> None:
    config = tomllib.loads((ROOT / "framework/fuzzyxai/pyproject.toml").read_text(encoding="utf-8"))
    excluded = set(config["tool"]["setuptools"]["packages"]["find"]["exclude"])
    assert {
        "fuzzyxai.ai_pre_review*",
        "fuzzyxai.ai_pre_review_final*",
        "fuzzyxai.final_closure*",
        "fuzzyxai.q1_final*",
        "fuzzyxai.q1_validation*",
        "fuzzyxai.strong_confirmatory*",
    } <= excluded


def test_final_validation_has_no_competing_system_runtime_values() -> None:
    validation = ROOT / "final_transparency_validation"
    for name in ("accept", "conflict", "reduction"):
        assert not (validation / f"golden_system_{name}" / "runtime_values.json").exists()
    assert not (validation / "architecture_matrix" / "matrix.json").exists()


def test_semantic_audit_does_not_draw_gamma_directly_into_rho() -> None:
    audit = (ROOT / "final_transparency_validation/semantic_audit.md").read_text(encoding="utf-8")
    assert "Gamma/u_M/Delta/I_pre -> rho" not in audit
    assert "rho_p/u_M/(1-I_pre)/Delta/chi_R -> rho" in audit
