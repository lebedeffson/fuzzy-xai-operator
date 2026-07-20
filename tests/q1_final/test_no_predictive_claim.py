from __future__ import annotations

from pathlib import Path


FORBIDDEN = (
    "critical rupture predicts errors",
    "critical rupture is a safety predictor",
    "critical rupture improves safety",
)


def test_q1_final_public_material_has_no_predictive_rupture_claim() -> None:
    root = Path(__file__).resolve().parents[2]
    targets = (
        root / "framework/fuzzyxai/fuzzyxai/q1_final",
        root / "research/q1_final",
        root / "study/q1_final",
    )
    violations = []
    for target in targets:
        for path in target.rglob("*"):
            if path.suffix not in {".py", ".md", ".yaml", ".json"} or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8").lower()
            violations.extend((path, phrase) for phrase in FORBIDDEN if phrase in text)
    assert not violations
