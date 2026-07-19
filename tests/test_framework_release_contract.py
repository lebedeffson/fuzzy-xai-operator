from __future__ import annotations

import ast
import json
from pathlib import Path

from fuzzyxai import FuzzyXAI
from fuzzyxai.schemas.validate import validate_payload


ROOT = Path(__file__).resolve().parents[1]


class CallableModel:
    def __call__(self, values):
        return [int(sum(row) > 0) for row in values]


def test_callable_model_returns_partial_honest_explanation() -> None:
    result = FuzzyXAI.wrap(CallableModel()).explain_one([1.0, -0.2], object_id="callable-1")
    assert result.action == "review"
    assert "model_rules_or_concepts" in result.view_model.trace["missing_evidence"]
    assert result.view_model.risk["rho"] is None


def test_view_model_schema_validation() -> None:
    result = FuzzyXAI.wrap(CallableModel()).explain_one([1.0, -0.2], object_id="callable-1")
    validation = validate_payload(result.to_dict(), "explanation_view_model")
    assert validation.valid, validation.errors


def test_explanation_is_deterministic_except_generation_time() -> None:
    first = FuzzyXAI.wrap(CallableModel()).explain_one([1.0, -0.2], object_id="same").to_dict()
    second = FuzzyXAI.wrap(CallableModel()).explain_one([1.0, -0.2], object_id="same").to_dict()
    first["trace"].pop("generated_at")
    second["trace"].pop("generated_at")
    assert first == second


def test_core_does_not_import_public_facade_or_presentation() -> None:
    forbidden = ("fuzzyxai.operators", "fuzzyxai.visual", "fuzzyxai.viz", "fuzzyxai.visualization")
    violations: list[str] = []
    for path in (ROOT / "framework/fuzzyxai/fuzzyxai/core").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(forbidden):
                violations.append(f"{path.name}:{node.lineno}:{node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(forbidden):
                        violations.append(f"{path.name}:{node.lineno}:{alias.name}")
    assert not violations, violations


def test_matlab_transport_surface_is_complete() -> None:
    matlab = ROOT / "framework/fuzzyxai/matlab/+fuzzyxai"
    expected = {
        "loadResult.m",
        "validateResult.m",
        "dashboard.m",
        "rulePlot.m",
        "membershipPlot.m",
        "trainingTrajectory.m",
        "similarCasesPlot.m",
        "provenancePlot.m",
        "explanationStory.m",
        "dataProfile.m",
    }
    assert expected <= {path.name for path in matlab.glob("*.m")}
    assert "schema_version must be 2.0" in (matlab / "validateResult.m").read_text(encoding="utf-8")


def test_site_is_quarantined_and_not_part_of_framework_tree() -> None:
    assert not (ROOT / "site/dubnaxai").exists()
    assert "archive/site-prototype-cab4018" in (ROOT / "site/README.md").read_text(encoding="utf-8")


def test_object_85_release_evidence_is_measured() -> None:
    payload = json.loads(
        (ROOT / "release_evidence/framework_completion/object_85_restoration_report.json").read_text(encoding="utf-8")
    )
    assert payload["forgetting_events"]
    assert payload["before_restoration"]["rare_subtype_recall"] == 0.0
    assert payload["after_restoration"]["rare_subtype_recall"] == 1.0
    assert payload["effect"]["test_accuracy"] < 0.0
