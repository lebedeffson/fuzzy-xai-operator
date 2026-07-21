from __future__ import annotations

from pathlib import Path

import pytest

from fuzzyxai.ai_pre_review import build_study_inputs


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session", autouse=True)
def generated_ai_pre_review_inputs() -> None:
    if not (ROOT / "study/ai_pre_review/master_explanation_log.jsonl").is_file():
        build_study_inputs(ROOT)
