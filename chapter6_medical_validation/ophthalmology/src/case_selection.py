from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def select_registered_cases(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    values = list(rows)
    if not values:
        raise ValueError("case selection requires frozen predictions")

    def choose(candidates: list[dict[str, Any]], key: str, *, reverse: bool) -> dict[str, Any] | None:
        if not candidates:
            return None
        return min(candidates, key=lambda row: ((-1 if reverse else 1) * float(row[key]), str(row["sample_id"])))

    correct = [row for row in values if int(row["prediction"]) == int(row["label"])]
    errors = [row for row in values if int(row["prediction"]) != int(row["label"])]
    both_xai = [row for row in values if row.get("explanation_disagreement") is not None]
    return {
        "A_confident_correct_grade0": choose([row for row in correct if int(row["label"]) == 0], "confidence", reverse=True),
        "B_confident_correct_referable": choose([row for row in correct if int(row["label"]) >= 2], "confidence", reverse=True),
        "C_boundary": choose(values, "top1_top2_margin", reverse=False),
        "D_high_confidence_error": choose(errors, "confidence", reverse=True),
        "E_lowest_technical_quality": choose(values, "technical_quality_score", reverse=False),
        "F_highest_explanation_disagreement": choose(both_xai, "explanation_disagreement", reverse=True),
    }
