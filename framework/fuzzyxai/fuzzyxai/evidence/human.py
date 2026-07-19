from __future__ import annotations

from typing import Sequence

from .contracts import ExplanationClaim, ExplanationGraph, HumanExplanation


def _grounded(claim: ExplanationClaim) -> str:
    return f"{claim.statement} [{claim.claim_id}]"


def compose_human_explanation(
    claims: Sequence[ExplanationClaim],
    graph: ExplanationGraph,
    *,
    action: str,
    level: str,
) -> HumanExplanation:
    """Compose text exclusively from auditable claims.

    Claim identifiers remain visible in every sentence, so a renderer can link
    prose back to evidence without reverse-engineering a free-text template.
    """

    if level not in {"user", "expert", "audit"}:
        raise ValueError("level must be user, expert, or audit")
    by_type: dict[str, list[ExplanationClaim]] = {}
    for claim in claims:
        by_type.setdefault(claim.claim_type, []).append(claim)

    prediction = by_type.get("prediction", [])[:1]
    action_claim = by_type.get("recommended_action", [])[:1]
    summary_claims = [*prediction, *action_claim]
    summary = " ".join(_grounded(claim) for claim in summary_claims)
    if action_claim:
        summary += f" Рекомендуемое действие: {action}. [{action_claim[0].claim_id}]"

    reasons = [
        *by_type.get("data_deviation", []),
        *by_type.get("model_rule", []),
        *by_type.get("class_concept", []),
    ]
    observed = [
        *by_type.get("data_quality", []),
        *by_type.get("first_learned", []),
        *by_type.get("prediction", []),
    ]
    lost = [
        *by_type.get("forgetting", []),
        *by_type.get("subgroup_averaging", []),
        *by_type.get("lost_rules", []),
    ]
    similar = by_type.get("similar_case", [])
    changes = by_type.get("counterfactual", [])
    trust = [*by_type.get("diagnostic", []), *by_type.get("recommended_action", [])]
    limitation_claims = [
        claim
        for claim in claims
        if claim.status in {"contested", "insufficient_evidence"} or claim.limitations
    ]

    if level == "user":
        reasons, observed, lost = reasons[:3], observed[:3], lost[:2]
        similar, changes, trust = similar[:2], changes[:2], trust[:2]
    elif level == "expert":
        reasons, observed = reasons[:7], observed[:7]
        similar, changes = similar[:5], changes[:5]

    limitations: list[str] = []
    for claim in limitation_claims:
        if claim.limitations:
            limitations.extend(f"{text} [{claim.claim_id}]" for text in claim.limitations)
        elif claim.status == "insufficient_evidence":
            limitations.append(_grounded(claim))

    def rendered(items: Sequence[ExplanationClaim]) -> list[str]:
        return [_grounded(item) for item in items]

    sections = {
        "summary": [claim.claim_id for claim in summary_claims],
        "main_reasons": [claim.claim_id for claim in reasons],
        "model_observed": [claim.claim_id for claim in observed],
        "lost_or_averaged": [claim.claim_id for claim in lost],
        "similar_cases": [claim.claim_id for claim in similar],
        "decision_changes": [claim.claim_id for claim in changes],
        "trust": [claim.claim_id for claim in trust],
        "limitations": [claim.claim_id for claim in limitation_claims],
    }
    trace = [] if level == "user" else [claim.claim_id for claim in claims]
    if level == "audit":
        trace.extend(node.node_id for node in graph.nodes)
    return HumanExplanation(
        level=level,
        summary=summary,
        main_reasons=rendered(reasons),
        model_observed=rendered(observed),
        lost_or_averaged=rendered(lost),
        similar_cases=rendered(similar),
        decision_changes=rendered(changes),
        trust=rendered(trust),
        limitations=list(dict.fromkeys(limitations)),
        recommended_action=action,
        evidence_trace=list(dict.fromkeys(trace)),
        claim_refs=sections,
    )


def explanation_to_text(explanation: HumanExplanation) -> str:
    """Render a claim-grounded explanation as readable Markdown text."""

    sections = [
        ("Итог", [explanation.summary]),
        ("Главные причины", explanation.main_reasons),
        ("Что модель увидела", explanation.model_observed),
        ("Что потеряно или усреднено", explanation.lost_or_averaged),
        ("Похожие случаи", explanation.similar_cases),
        ("Что изменило бы решение", explanation.decision_changes),
        ("Доверие", explanation.trust),
        ("Ограничения", explanation.limitations),
        ("Доказательный след", explanation.evidence_trace),
    ]
    lines: list[str] = []
    for title, values in sections:
        if not values:
            continue
        lines.append(f"## {title}")
        lines.extend(f"- {value}" for value in values)
        lines.append("")
    return "\n".join(lines).strip() + "\n"
