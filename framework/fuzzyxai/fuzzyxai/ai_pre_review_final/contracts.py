"""Public and private boundaries for the final blind review study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


DENYLIST = (
    "is_correct",
    "true_label",
    "wrong_prediction",
    "correct_prediction",
    "ground_truth",
    "expected_action",
    "action_accept",
    "action_short_review",
    "action_full_review",
    "action_block",
    "known_contradictions",
    "known_unsupported_claims",
    "structural_rupture",
    "no_structural_rupture",
    "missing_provenance",
    "complete_provenance",
    "method_name",
    "method_identity",
    "original_stratum",
    "hidden_rupture_type",
    "answer_key",
    "controlled_condition",
    "fuzzyxai",
    "shap",
    "lime",
    "anchors",
    "rulefit",
    "grad-cam",
    "integrated gradients",
)
METHODS = ("strong_simple_baseline", "selective_observer", "full_operator_explanation")


class FinalStudyError(RuntimeError):
    """Raised when a leakage or evidence-integrity boundary is violated."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: bytes | str) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise FinalStudyError(f"{path}:{line_number}: row is not an object")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(dict(row)) + "\n" for row in rows), encoding="utf-8")
