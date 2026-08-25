from __future__ import annotations

import re
from dataclasses import dataclass

from fuzzyxai.evidence.contracts import AtomicClaim
from fuzzyxai.verbalization.atomic_claims import (
    combined_allowed_entities,
    combined_allowed_numbers,
    combined_source_text,
)

_NUMBER_TOKEN = re.compile(r"-?\d[\d\s]*(?:[.,]\d+)?\s*%?(?:[eE][+-]?\d+)?")
_EPSILON = 1e-6

# Words asserting causation or certainty that the source claims did not
# license. These are rejected in generated text unless the exact word (not a
# paraphrase — this is a surface check, not semantic entailment) already
# appears in the source claims.
CAUSAL_CERTAINTY_WORDS = (
    "доказывает",
    "доказано",
    "гарантирует",
    "гарантированно",
    "вызывает",
    "точно",
    "определённо",
    "несомненно",
    "обязательно",
    "proves",
    "guarantees",
    "causes",
    "certainly",
    "definitely",
    "undeniably",
)

MAX_REWRITE_CHARS = 2000


@dataclass(frozen=True)
class GuardResult:
    passed: bool
    checks: tuple[str, ...]  # every check name that was evaluated, in order
    failed_check: str | None
    reason: str | None


def _normalize_number(token: str) -> float | None:
    cleaned = token.strip().replace(" ", "").replace(",", ".")
    is_percent = cleaned.endswith("%")
    cleaned = cleaned.rstrip("%")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value / 100.0 if is_percent else value


def _numbers_equivalent(candidate: float, allowed: set[float]) -> bool:
    for value in allowed:
        if abs(candidate - value) < _EPSILON:
            return True
        # percent/fraction equivalence: 50 vs 0.5, 0.5 vs 50
        if abs(candidate - value * 100) < _EPSILON or abs(candidate - value / 100) < _EPSILON:
            return True
    return False


def check_no_new_numbers(generated_text: str, claims: tuple[AtomicClaim, ...]) -> GuardResult:
    allowed_raw = combined_allowed_numbers(claims)
    allowed_values = {value for token in allowed_raw if (value := _normalize_number(token)) is not None}
    for match in _NUMBER_TOKEN.findall(generated_text):
        value = _normalize_number(match)
        if value is None:
            continue
        if not _numbers_equivalent(value, allowed_values):
            return GuardResult(False, ("no_new_numbers",), "no_new_numbers", f"generated text contains number '{match.strip()}' not present in the source claims")
    return GuardResult(True, ("no_new_numbers",), None, None)


# Deliberately narrow: matches things that look like feature/class/technical
# identifiers (snake_case, ALL-CAPS acronyms, alphanumeric codes) — not
# ordinary prose words. Natural-language paraphrase changes word forms
# constantly (Russian inflection especially), so checking *every* word
# against the source would reject almost any real rephrasing; checking only
# identifier-shaped tokens targets the actual threat this guard is for —
# a fabricated feature/class/entity name — without that false-positive rate.
_TECHNICAL_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9]*(?:_[A-Za-zА-Яа-яЁё0-9]+)+|\b[A-Z]{2,}\d*\b")


def check_no_new_entities(generated_text: str, claims: tuple[AtomicClaim, ...]) -> GuardResult:
    allowed = combined_allowed_entities(claims)
    generated_tokens = {token.lower() for token in _TECHNICAL_TOKEN.findall(generated_text)}
    unknown = generated_tokens - allowed
    if unknown:
        return GuardResult(False, ("no_new_entities",), "no_new_entities", f"generated text introduces unfamiliar feature/class/entity names not in the source claims: {sorted(unknown)[:5]}")
    return GuardResult(True, ("no_new_entities",), None, None)


def check_no_unlicensed_certainty_language(generated_text: str, claims: tuple[AtomicClaim, ...]) -> GuardResult:
    lowered = generated_text.lower()
    source = combined_source_text(claims).lower()
    for word in CAUSAL_CERTAINTY_WORDS:
        if word in lowered and word not in source:
            return GuardResult(False, ("no_unlicensed_certainty_language",), "no_unlicensed_certainty_language", f"generated text uses '{word}', a stronger causal/certainty claim than the source evidence supports")
    return GuardResult(True, ("no_unlicensed_certainty_language",), None, None)


def check_non_empty_and_bounded(generated_text: str, *, max_chars: int = MAX_REWRITE_CHARS) -> GuardResult:
    if not generated_text.strip():
        return GuardResult(False, ("non_empty_and_bounded",), "non_empty_and_bounded", "generated text is empty")
    if len(generated_text) > max_chars:
        return GuardResult(False, ("non_empty_and_bounded",), "non_empty_and_bounded", f"generated text exceeds {max_chars} characters")
    return GuardResult(True, ("non_empty_and_bounded",), None, None)


def run_rewrite_guards(generated_text: str, claims: tuple[AtomicClaim, ...]) -> GuardResult:
    """Run every rewrite-mode guard in order; stop and report the first failure.

    These are *surface* checks — no new number, no new entity/name, no
    unlicensed causal/certainty language, non-empty and length-bounded. They
    do not, and cannot, prove that every sentence is a valid semantic
    entailment of its cited claim; that distinction must stay visible to
    callers (see ``VerbalizationResult.status`` and the module docstring).
    """

    checks: list[str] = []
    for result in (
        check_non_empty_and_bounded(generated_text),
        check_no_new_numbers(generated_text, claims),
        check_no_new_entities(generated_text, claims),
        check_no_unlicensed_certainty_language(generated_text, claims),
    ):
        checks.extend(result.checks)
        if not result.passed:
            return GuardResult(False, tuple(checks), result.failed_check, result.reason)
    return GuardResult(True, tuple(checks), None, None)
