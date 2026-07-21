from __future__ import annotations

from pathlib import Path

from fuzzyxai.ai_pre_review import build_study_inputs


ROOT = Path(__file__).resolve().parents[2]


def pytest_sessionstart(session) -> None:
    del session
    if not (ROOT / "study/ai_pre_review/master_explanation_log.jsonl").is_file():
        build_study_inputs(ROOT)
