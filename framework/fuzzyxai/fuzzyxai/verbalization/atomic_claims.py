from __future__ import annotations

import re
from typing import Literal

from fuzzyxai.evidence.contracts import AtomicClaim, HumanExplanation, HumanStatement, ReasonStatement

_ClaimKind = Literal["decision", "reason", "concern", "reliability", "action"]

_NUMBER_PATTERN = re.compile(r"-?\d+(?:[.,]\d+)?%?")
WORD_PATTERN = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_STOPWORDS = frozenset(
    ["и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "her", "all", "also", "his", "have", "from", "they", "were", "been", "that", "this", "with", "your", "which", "their", "would", "there", "the", "a", "an", "is", "are", "was", "be", "to", "of", "for", "on", "it", "as", "at", "by"]
)


def _numbers_in(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_NUMBER_PATTERN.findall(text)))


def _entities_in(text: str) -> tuple[str, ...]:
    words = (word.lower() for word in WORD_PATTERN.findall(text))
    return tuple(dict.fromkeys(word for word in words if word not in _STOPWORDS))


def _claim(claim_id: str, kind: _ClaimKind, statement: HumanStatement) -> AtomicClaim:
    text = statement.explanation
    extra_entities: tuple[str, ...] = ()
    direction = "neutral"
    if isinstance(statement, ReasonStatement):
        extra_entities = (statement.subject_label.lower(),)
        direction = statement.effect_direction
    return AtomicClaim(
        claim_id=claim_id,
        kind=kind,
        subject=statement.title,
        canonical_text=text,
        allowed_numbers=_numbers_in(text),
        # the subject itself must be in the allowed-entity closure too, so a
        # rewrite-mode backend echoing "Feature 22" back isn't flagged as a
        # fabricated new entity.
        allowed_entities=tuple(dict.fromkeys((*extra_entities, *_entities_in(text), *_entities_in(statement.title)))),
        direction=direction,
        # Traces this atomic claim back to the ExplanationClaim(s) it was
        # built from — the same provenance already carried by every
        # HumanStatement, just also exposed at the verbalizer-facing layer.
        source_claim_ids=tuple(statement.claim_refs),
    )


def extract_atomic_claims(explanation: HumanExplanation) -> tuple[AtomicClaim, ...]:
    """Turn an already-verified HumanExplanation into the closed set of facts a verbalizer may use.

    This is the *only* thing a verbalization backend ever sees — never the
    raw prediction, evidence graph, or ExplanationClaim internals directly.
    Each ``AtomicClaim`` records its own ``allowed_numbers``/``allowed_entities``
    closure, extracted from its own canonical text (plus, for reasons, the
    subject label) — this closure is what the grounding guard checks new
    output against.
    """

    claims: list[AtomicClaim] = [_claim("decision-0", "decision", explanation.decision)]
    reasons = list(explanation.main_reasons)
    claims.extend(_claim(f"reason-{index}", "reason", reason) for index, reason in enumerate(reasons))
    claims.extend(_claim(f"concern-{index}", "concern", concern) for index, concern in enumerate(explanation.concerns))
    # summary() folds a similarity digest into its text (P1.1), but a
    # similar-case reason only reaches a verbalizer when it happens to rank
    # into main_reasons/concerns — a strict backend otherwise has no
    # similarity claim_id to select, so it silently drops the exemplar
    # summary() shows. Add any not-already-covered similar-case statement as
    # its own reason so it's always selectable.
    already_covered = {ref for claim in claims for ref in claim.source_claim_ids}
    for offset, similar in enumerate(explanation.details.similar_cases):
        if any(ref in already_covered for ref in similar.claim_refs):
            continue
        claims.append(_claim(f"reason-{len(reasons) + offset}", "reason", similar))
    claims.append(_claim("reliability-0", "reliability", explanation.reliability))
    claims.append(_claim("action-0", "action", explanation.recommended_action))
    return tuple(claims)


def combined_allowed_numbers(claims: tuple[AtomicClaim, ...]) -> frozenset[str]:
    return frozenset(number for claim in claims for number in claim.allowed_numbers)


def combined_allowed_entities(claims: tuple[AtomicClaim, ...]) -> frozenset[str]:
    return frozenset(entity for claim in claims for entity in claim.allowed_entities)


def combined_source_text(claims: tuple[AtomicClaim, ...]) -> str:
    return " ".join(claim.canonical_text for claim in claims)
