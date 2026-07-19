from __future__ import annotations

from typing import Any, Mapping

from .contracts import ExplanationEvidence, ExplanationGraph, HumanExplanation


def compose_human_explanation(
    evidence: ExplanationEvidence,
    graph: ExplanationGraph,
    *,
    prediction: Mapping[str, Any],
    action: str,
    level: str,
) -> HumanExplanation:
    """Generate user/expert/audit text using facts present in the graph only."""

    if level not in {"user", "expert", "audit"}:
        raise ValueError("level must be user, expert, or audit")
    score = prediction.get("score")
    predictions = prediction.get("predictions")
    summary = f"Модель сформировала прогноз {predictions}."
    if isinstance(score, (int, float)):
        summary += f" Максимальный модельный балл равен {score:.3f}."
    summary += f" Рекомендуемое действие: {action}."

    reasons: list[str] = []
    observed: list[str] = []
    lost: list[str] = []
    similar_text: list[str] = []
    changes: list[str] = []
    trust: list[str] = []
    limitations: list[str] = []
    trace: list[str] = []

    for data in evidence.data:
        if data.anomaly_labels:
            reasons.append(
                f"Объект {data.object_id} отличается от медианного профиля по признакам {', '.join(data.anomaly_labels)}."
            )
        observed.append(f"Качество входных данных объекта {data.object_id}: {data.data_quality:.3f}.")
        limitations.extend(data.warnings)
        trace.extend(data.evidence_refs)
    for item in evidence.training:
        if item.first_learned_epoch is not None:
            observed.append(f"Объект {item.object_id} впервые устойчиво распознан на эпохе {item.first_learned_epoch}.")
        if item.forgetting_events:
            lost.append(f"Для объекта {item.object_id} обнаружены события забывания на эпохах {list(item.forgetting_events)}.")
    for subgroup in evidence.subgroups:
        if subgroup.averaged:
            lost.append(
                f"При росте общей метрики на {subgroup.global_metric_change:+.3f} метрика подгруппы {subgroup.subgroup_id} изменилась на {subgroup.subgroup_metric_change:+.3f}."
            )
            if subgroup.disappeared_rules:
                lost.append(f"Перестали наблюдаться правила: {', '.join(subgroup.disappeared_rules)}.")
        limitations.extend(subgroup.limitations)
    primary_rules = [rule for rule in evidence.rules if rule.is_primary][:7]
    for rule in primary_rules:
        kind = "нативное" if rule.native else "суррогатное"
        importance = "не измерена" if rule.importance is None else f"{rule.importance:.3f}"
        reasons.append(f"Правило {rule.rule_id} ({kind}): {rule.human_text}; значимость {importance}.")
        trace.extend(rule.evidence_refs)
    for concept in evidence.concepts:
        observed.append(concept.human_description)
        limitations.extend(concept.limitations)
    for case in evidence.similar_cases:
        similar_text.append(
            f"Объект {case.reference_object_id}: сходство {case.similarity_score:.3f} по методу {case.similarity_method}; сравнивались {case.compared_representation}."
        )
        limitations.extend(case.limitations)
    for counterfactual in evidence.counterfactuals:
        change = (
            f"признаков {dict(counterfactual.changed_features)}"
            if counterfactual.changed_features
            else f"правил {list(counterfactual.changed_rules)}"
        )
        changes.append(
            f"Изменение {change} переводит прогноз из {counterfactual.source_prediction} в {counterfactual.target_prediction}; наблюдаемый эффект {counterfactual.observed_effect}."
        )
        limitations.extend(counterfactual.limitations)
        trace.extend(counterfactual.evidence_refs)
    if evidence.missing:
        limitations.append("Недоступны доказательства: " + ", ".join(evidence.missing) + ".")
        trust.append("Неполный evidence не позволяет автоматически принимать решение.")
    else:
        trust.append("Все заявленные факты связаны с узлами ExplanationGraph.")

    if level == "user":
        reasons = reasons[:3]
        observed = observed[:3]
        lost = lost[:2]
        similar_text = similar_text[:2]
        changes = changes[:2]
        trace = []
    elif level == "expert":
        trace = sorted(set(trace))[:12]
    else:
        trace = [node.node_id for node in graph.nodes] + sorted(set(trace))
    return HumanExplanation(
        level=level,
        summary=summary,
        main_reasons=reasons,
        model_observed=observed,
        lost_or_averaged=lost,
        similar_cases=similar_text,
        decision_changes=changes,
        trust=trust,
        limitations=list(dict.fromkeys(limitations)),
        recommended_action=action,
        evidence_trace=list(dict.fromkeys(trace)),
    )


def explanation_to_text(explanation: HumanExplanation) -> str:
    """Render a structured explanation as readable Markdown text."""

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
