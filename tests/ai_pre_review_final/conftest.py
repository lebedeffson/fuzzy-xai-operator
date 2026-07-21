from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session", autouse=True)
def generated_public_ai_pre_review_bundle() -> None:
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    archive = (
        ROOT
        / "release_artifacts/ai_pre_review_final"
        / f"fuzzyxai-ai-pre-review-final-bundles-{commit}.zip"
    )
    if not archive.is_file():
        subprocess.run(
            [sys.executable, "scripts/ai_pre_review_final/build_public_bundle.py"],
            cwd=ROOT,
            check=True,
        )
