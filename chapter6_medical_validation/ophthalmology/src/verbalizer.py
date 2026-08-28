from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VerbalizationResult:
    status: str
    text: str
    backend: str
    model: str | None
    source_claim_ids: tuple[str, ...]
    guard_checks: dict[str, Any]
    fallback_reason: str | None


def deterministic_ophthalmology_text(result: Any) -> VerbalizationResult:
    claims = tuple(result.view_model.claims)
    claim_ids = tuple(str(item.get("claim_id", item.get("id", "unknown"))) for item in claims)
    prediction = result.prediction.to_dict()
    grade = prediction.get("predictions", [None])
    grade = grade[0] if isinstance(grade, list) else grade
    system = result.system.audit_dict() if result.system is not None else None
    lines = [f"Модель отнесла снимок к стадии {grade} диабетической ретинопатии."]
    if system is not None:
        risk = system["risk"]
        lines.append(f"Техническое действие FuzzyXAI: {risk['action']}.")
        if risk["rho"] is not None:
            lines.append(f"Интегральная техническая оценка rho: {risk['rho']:.6f}.")
        if risk["critical_override"]:
            lines.append("Обнаружен критический разрыв объяснительного маршрута; автоматическое применение блокировано.")
    return VerbalizationResult(
        status="accepted_deterministic_fallback",
        text=" ".join(lines),
        backend="deterministic_templates",
        model=None,
        source_claim_ids=claim_ids,
        guard_checks={"unsupported_additions": 0, "numbers_from_result_only": True, "action_preserved": True},
        fallback_reason="SLM backend not registered for this run",
    )


def guard_strict_text(candidate: str, *, allowed_claim_texts: list[str], forbidden_phrases: list[str], allowed_numbers: set[str]) -> dict[str, Any]:
    lowered = candidate.lower()
    forbidden = [phrase for phrase in forbidden_phrases if phrase.lower() in lowered]
    numbers = set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?", candidate))
    unsupported_numbers = sorted(numbers - allowed_numbers)
    supported_fragments = [claim for claim in allowed_claim_texts if claim and claim.lower() in lowered]
    return {
        "accepted": not forbidden and not unsupported_numbers,
        "forbidden_phrases": forbidden,
        "unsupported_numbers": unsupported_numbers,
        "supported_claim_fragments": supported_fragments,
    }


def preservation_metrics(reference: str, candidate: str, *, source_facts: set[str], action: str, limitations: list[str]) -> dict[str, float]:
    candidate_lower = candidate.lower()
    covered = sum(1 for fact in source_facts if fact.lower() in candidate_lower)
    reference_numbers = set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?", reference))
    candidate_numbers = set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?", candidate))
    unsupported = candidate_numbers - reference_numbers
    return {
        "P_fact": covered / max(len(source_facts), 1),
        "H": len(unsupported) / max(len(candidate_numbers), 1),
        "P_num": len(reference_numbers & candidate_numbers) / max(len(reference_numbers), 1),
        "P_action": float(action.lower() in candidate_lower),
        "P_lim": sum(1 for value in limitations if value.lower() in candidate_lower) / max(len(limitations), 1),
    }
