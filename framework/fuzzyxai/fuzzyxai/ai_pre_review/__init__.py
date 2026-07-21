"""Evidence-first tooling for blind AI pre-review and human confirmation."""

from .contracts import (
    AI_RUN_IDS,
    CRITICAL_FLAGS,
    SCORE_KEYS,
    StudyBoundaryError,
    sha256_file,
)
from .generator import build_study_inputs
from .review_io import aggregate_ai_reviews, compare_ai_human, validate_ai_review_directory, validate_human_review_directory

__all__ = [
    "AI_RUN_IDS",
    "CRITICAL_FLAGS",
    "SCORE_KEYS",
    "StudyBoundaryError",
    "aggregate_ai_reviews",
    "build_study_inputs",
    "compare_ai_human",
    "sha256_file",
    "validate_ai_review_directory",
    "validate_human_review_directory",
]
