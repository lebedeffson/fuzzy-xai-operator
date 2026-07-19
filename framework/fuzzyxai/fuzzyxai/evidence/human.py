from __future__ import annotations

import ast
import re
from typing import Any, Mapping, Sequence, cast

from .contracts import (
    ActionStatement,
    AudienceName,
    AudienceProfile,
    ChangeStatement,
    ConcernStatement,
    DecisionStatement,
    ExplanationClaim,
    ExplanationDetails,
    ExplanationEvidence,
    ExplanationGraph,
    HumanExplanation,
    HumanStatement,
    ReasonStatement,
    ReliabilityStatement,
)


AUDIENCE_PROFILES: dict[AudienceName, AudienceProfile] = {
    "domain_user": AudienceProfile("domain_user", 3, 2, 1, False, False),
    "ml_engineer": AudienceProfile("ml_engineer", 7, 7, 5, True, True),
    "researcher": AudienceProfile("researcher", 7, 7, 5, True, True),
    "auditor": AudienceProfile("auditor", 1000, 1000, 1000, True, True),
}

TECHNICAL_TERMS = (
    re.compile(r"\bR\d+\b", re.IGNORECASE),
    re.compile(r"\bS\d+\b", re.IGNORECASE),
    re.compile(r"\bE[0-5]\b", re.IGNORECASE),
    re.compile(r"\b(?:gamma|delta|rho|chi[_a-z]*|claim(?:_id)?)\b", re.IGNORECASE),
    re.compile(r"\b(?:defer_to_human|audit_report|insufficient_evidence)\b", re.IGNORECASE),
)

_TYPE_WEIGHT = {
    "recommended_action": 1.00,
    "counterfactual": 0.98,
    "forgetting": 0.97,
    "subgroup_averaging": 0.96,
    "lost_rules": 0.95,
    "diagnostic": 0.93,
    "feature_contribution": 0.92,
    "data_deviation": 0.88,
    "model_rule": 0.84,
    "class_concept": 0.80,
    "similar_case": 0.72,
    "data_quality": 0.45,
    "prediction": 1.00,
}

_ACTION_TEXT = {
    "accept": ("Использовать с обычным контролем", "Доступные данные не выявили ограничения, требующие отдельной проверки."),
    "review": ("Проверить специалистом", "Результат недостаточно надёжен для автоматического решения."),
    "audit": ("Провести дополнительную проверку", "Перед применением результата необходимо проверить доказательный след."),
    "audit_report": ("Провести дополнительную проверку", "Перед применением результата необходимо проверить доказательный след."),
    "block": ("Не применять автоматически", "Обнаружено критическое ограничение; решение должно быть остановлено и проверено."),
    "defer_to_human": ("Передать специалисту", "Автоматическое решение ограничено; требуется предметная проверка."),
    "insufficient_evidence": ("Собрать недостающие данные", "Доступных сведений недостаточно для автоматического решения."),
}

_REPRESENTATION_TEXT = {
    "normalized tabular feature vector": "нормализованные значения признаков",
    "model embedding vector": "внутреннее представление модели",
    "segmentation masks": "сегментационные маски",
}


def audience_profile(name: str) -> AudienceProfile:
    aliases = {"user": "domain_user", "expert": "ml_engineer", "audit": "auditor"}
    normalized = aliases.get(name, name)
    if normalized not in AUDIENCE_PROFILES:
        raise ValueError("audience must be domain_user, ml_engineer, researcher, or auditor")
    return AUDIENCE_PROFILES[normalized]


def rank_human_claims(
    claims: Sequence[ExplanationClaim],
    *,
    domain_language: Mapping[str, Any] | None = None,
) -> list[ExplanationClaim]:
    """Rank claims by decision relevance rather than raw metric magnitude."""

    domain = dict(domain_language or {})
    domain_features = domain.get("features", {}) if isinstance(domain.get("features", {}), Mapping) else {}

    def score(claim: ExplanationClaim) -> tuple[float, str]:
        importance = _TYPE_WEIGHT.get(claim.claim_type, 0.50)
        understandable = 0.85 if claim.claim_type not in {"diagnostic", "missing_channel"} else 0.55
        confirmation = {"supported": 1.0, "contested": 0.72, "insufficient_evidence": 0.60, "not_applicable": 0.20}[claim.evidence_status]
        domain_significance = 1.0 if claim.subject_id in domain_features or claim.severity == "critical" else 0.82
        action_influence = 1.0 if claim.claim_type in {"recommended_action", "counterfactual", "diagnostic", "forgetting", "lost_rules"} else 0.78
        measured = 0.75 + 0.25 * (claim.strength if claim.strength is not None else 0.5)
        return importance * understandable * confirmation * domain_significance * action_influence * measured, claim.claim_id

    return sorted(claims, key=score, reverse=True)


def _refs(claims: Sequence[ExplanationClaim]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(dict.fromkeys(claim.claim_id for claim in claims)),
        tuple(dict.fromkeys(str(ref) for claim in claims for ref in claim.evidence_refs)),
    )


def _domain_section(domain: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = domain.get(name, {})
    return value if isinstance(value, Mapping) else {}


def _domain_entry(section: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = section.get(key)
    if value is None:
        value = next((item for candidate, item in section.items() if str(candidate) == str(key)), {})
    return value if isinstance(value, Mapping) else {}


def _prediction_value(subject: str) -> str:
    try:
        parsed = ast.literal_eval(subject)
        if isinstance(parsed, (list, tuple)) and parsed:
            return str(parsed[0])
    except (ValueError, SyntaxError):
        pass
    return subject


def _clean_internal_identifiers(text: str) -> str:
    value = re.sub(r"\bR[\w-]*\b", "специальное правило", text, flags=re.IGNORECASE)
    value = re.sub(r"\bS\d+\b", "редкая группа", value, flags=re.IGNORECASE)
    value = re.sub(r"\bE[0-5]\b", "доступный уровень объяснения", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:gamma|delta|rho|chi[_a-z]*)\b", "технический показатель", value, flags=re.IGNORECASE)
    for code, (label, _) in _ACTION_TEXT.items():
        value = value.replace(code, label.lower())
    return value


def _decision_statement(claim: ExplanationClaim, domain: Mapping[str, Any], technical: bool) -> DecisionStatement:
    class_id = _prediction_value(claim.subject_id)
    class_entry = _domain_entry(_domain_section(domain, "classes"), class_id)
    label = str(class_entry.get("label", f"класс {class_id}"))
    text = f"Модель определила: {label}."
    if technical and claim.metric_value is not None:
        text += f" Модельный балл: {claim.metric_value:.3f}."
    claim_refs, evidence_refs = _refs((claim,))
    title = label[:1].upper() + label[1:]
    return DecisionStatement(title, text, claim_refs, evidence_refs)


def _feature_profile(evidence: ExplanationEvidence, feature: str) -> tuple[float | None, Mapping[str, float | None], Mapping[str, float | None]]:
    for item in evidence.data:
        if feature not in item.feature_names:
            continue
        index = list(item.feature_names).index(feature)
        raw = item.raw_values[index]
        value = float(raw) if isinstance(raw, (int, float)) else None
        return value, item.reference_profiles.get(feature, {}), item.subgroup_profiles.get(feature, {})
    return None, {}, {}


def _feature_reason(
    claim: ExplanationClaim,
    evidence: ExplanationEvidence,
    domain: Mapping[str, Any],
    technical: bool,
) -> ReasonStatement:
    entry = _domain_entry(_domain_section(domain, "features"), claim.subject_id)
    label = str(entry.get("label", claim.subject_id.replace("_", " ")))
    value, profile, subgroup = _feature_profile(evidence, claim.subject_id)
    percentile = profile.get("percentile")
    median = profile.get("median")
    if isinstance(percentile, (int, float)):
        if percentile >= 50:
            comparison = f"Значение выше, чем у {percentile:.0f}% объектов референсной выборки."
            domain_text = str(entry.get("high_text", ""))
        else:
            comparison = f"Значение ниже, чем у {100 - percentile:.0f}% объектов референсной выборки."
            domain_text = str(entry.get("low_text", ""))
    else:
        comparison = "Признак входит в число наиболее важных для текущего решения."
        domain_text = ""
    if value is not None and subgroup:
        low, high = subgroup.get("q05"), subgroup.get("q95")
        overall_low, overall_high = profile.get("q05"), profile.get("q95")
        if (
            isinstance(low, (int, float))
            and isinstance(high, (int, float))
            and isinstance(overall_low, (int, float))
            and isinstance(overall_high, (int, float))
            and low <= value <= high
            and not overall_low <= value <= overall_high
        ):
            comparison = "Значение необычно для всей выборки, но типично для редкой группы, к которой относится объект."
    effect = "поддерживает" if claim.effect == "favorable" else "противоречит"
    explanation = " ".join(part for part in (comparison, domain_text, f"Этот признак {effect} текущему решению.") if part)
    if technical and claim.metric_value is not None:
        explanation += f" Локальный вклад: {claim.metric_value:+.3f}; медиана: {median}."
    claim_refs, evidence_refs = _refs((claim,))
    return ReasonStatement(label.capitalize(), explanation, claim_refs, evidence_refs)


def _rule_reason(claim: ExplanationClaim, evidence: ExplanationEvidence, technical: bool) -> ReasonStatement:
    rule = next((item for item in evidence.rules if item.rule_id == claim.subject_id), None)
    title = f"Правило {claim.subject_id}" if technical else "Закономерность, найденная моделью"
    if technical:
        description = rule.human_text if rule else claim.statement
    elif rule and re.search(r"[А-Яа-яЁё]", rule.human_text):
        description = _clean_internal_identifiers(rule.human_text)
    else:
        description = "Подтверждённая закономерность связывает значения признаков с выбранным результатом и"
    explanation = f"{description.rstrip('.')} поддерживает текущее решение."
    if technical and rule and rule.importance is not None:
        explanation += f" Измеренная значимость: {rule.importance:.3f}."
    claim_refs, evidence_refs = _refs((claim,))
    return ReasonStatement(title, explanation, claim_refs, evidence_refs)


def _concept_reason(
    claim: ExplanationClaim,
    evidence: ExplanationEvidence,
    domain: Mapping[str, Any],
    technical: bool,
) -> ReasonStatement:
    concept = next((item for item in evidence.concepts if item.class_id == claim.subject_id), None)
    class_entry = _domain_entry(_domain_section(domain, "classes"), claim.subject_id)
    label = str(class_entry.get("label", concept.class_name if concept else claim.subject_id))
    if technical:
        explanation = concept.human_description if concept else claim.statement
    elif concept and re.search(r"[А-Яа-яЁё]", concept.human_description):
        explanation = _clean_internal_identifiers(concept.human_description)
    else:
        explanation = (
            f"По сравнению с референсной выборкой объект близок к типичному профилю группы «{label}». "
            "Такое сходство поддерживает результат, но не заменяет проверку ограничений."
        )
    claim_refs, evidence_refs = _refs((claim,))
    return ReasonStatement(f"Типичный профиль класса {label}", explanation, claim_refs, evidence_refs)


def _similar_statement(claim: ExplanationClaim, evidence: ExplanationEvidence, technical: bool) -> HumanStatement:
    refs = set(claim.evidence_refs)
    candidates = [
        item
        for item in evidence.similar_cases
        if item.query_object_id == claim.subject_id
        and f"similar:{item.query_object_id}:{item.reference_object_id}" in refs
    ]
    case = next(
        (
            item
            for item in candidates
            if claim.comparison_baseline == item.compared_representation
            and (claim.metric_value is None or abs(claim.metric_value - item.similarity_score) < 1e-9)
        ),
        candidates[0] if candidates else None,
    )
    if case is None:
        claim_refs, evidence_refs = _refs((claim,))
        return HumanStatement("Похожий случай", _clean_internal_identifiers(claim.statement), claim_refs, evidence_refs)
    representation = case.compared_representation.lower()
    image_similarity = any(token in representation for token in ("mask", "segmentation", "image", "region"))
    if image_similarity:
        explanation = (
            f"Модель нашла обучающий пример с похожей выделенной областью. "
            f"Контуры областей перекрываются на {case.similarity_score * 100:.0f}%. "
            "Этот показатель относится к геометрии выделенных областей и не является вероятностью одинакового диагноза."
        )
        title = "Похожая область изображения"
    else:
        representation_text = _REPRESENTATION_TEXT.get(case.compared_representation.lower(), case.compared_representation)
        if case.is_counterexample or claim.effect == "adverse":
            explanation = (
                f"Найден близкий контрпример; сходство рассчитано по представлению «{representation_text}». "
                "Он показывает, что отдельное сходство встречается и у другого результата, поэтому ограничивает доверие."
            )
            title = "Похожий контрпример"
        else:
            explanation = (
                f"Найден похожий обучающий случай; сходство рассчитано по представлению «{representation_text}». "
                "Сходство поддерживает результат, но само по себе не доказывает одинаковый исход."
            )
            title = "Похожий обучающий случай"
    if technical:
        explanation += f" Метод: {case.similarity_method}; значение: {case.similarity_score:.3f}; объект: {case.reference_object_id}."
    claim_refs, evidence_refs = _refs((claim,))
    return HumanStatement(title, explanation, claim_refs, evidence_refs)


def _training_concern(claims: Sequence[ExplanationClaim], evidence: ExplanationEvidence, technical: bool) -> ConcernStatement:
    forgetting = next((claim for claim in claims if claim.claim_type == "forgetting"), None)
    trace = next((item for item in evidence.training if forgetting and item.object_id == forgetting.subject_id), None)
    label_loss_event = None
    if trace:
        for previous, current in zip(trace.epoch_metrics, trace.epoch_metrics[1:]):
            if bool(previous.get("correct", False)) and not bool(current.get("correct", False)):
                label_loss_event = int(current.get("epoch", 0))
                break
    event = label_loss_event or (trace.forgetting_events[0] if trace and trace.forgetting_events else None)
    object_id = forgetting.subject_id if forgetting else "рассматриваемый объект"
    if event is not None:
        explanation = (
            f"Модель сначала научилась правильно распознавать объект №{object_id}. "
            f"После {event}-го этапа обучения она снова начала ошибаться и стала хуже различать редкие случаи этого типа."
        )
    else:
        explanation = "Во время обучения модель стала хуже распознавать редкую группу, хотя общая метрика не показывала эту проблему."
    subgroup = next((item for item in evidence.subgroups if item.averaged), None)
    if subgroup and subgroup.global_metric_change > 0 and subgroup.subgroup_metric_change < 0:
        explanation += " Общий результат модели при этом улучшился, поэтому проблема не была видна по общей метрике."
    if technical:
        disappeared = [rule for item in evidence.subgroups for rule in item.disappeared_rules]
        if disappeared:
            explanation += f" Ослабленные правила: {', '.join(disappeared)}."
    claim_refs, evidence_refs = _refs(claims)
    return ConcernStatement("Редкий тип стал распознаваться хуже", explanation, claim_refs, evidence_refs)


def _generic_concern(claim: ExplanationClaim, technical: bool) -> ConcernStatement:
    if claim.claim_type == "data_deviation":
        text = "Объект отличается от большинства наблюдений. Это может быть редкий случай, а не ошибка данных, поэтому требуется предметная проверка."
        title = "Необычный объект"
    elif claim.claim_type in {"diagnostic", "missing_channel"}:
        text = "Для части проверки не хватает подтверждённых данных, поэтому автоматическое применение ограничено."
        title = "Не все проверки подтверждены"
    else:
        text = _clean_internal_identifiers(claim.statement)
        title = "Ограничение"
    if technical:
        text += f" Источник: {claim.claim_id}; статус: {claim.evidence_status}."
    claim_refs, evidence_refs = _refs((claim,))
    return ConcernStatement(title, text, claim_refs, evidence_refs)


def _change_statement(
    claim: ExplanationClaim,
    evidence: ExplanationEvidence,
    domain: Mapping[str, Any],
    technical: bool,
) -> ChangeStatement:
    item = next((value for value in evidence.counterfactuals if str(value.source_prediction) == claim.subject_id), None)
    if item and item.changed_rules and not item.changed_features:
        text = "Проверка показала, что восстановление специального правила улучшает распознавание редкой группы, даже если общая точность почти не меняется."
        title = "Восстановить знание о редкой группе"
    elif item and item.changed_features:
        feature_section = _domain_section(domain, "features")
        feature_labels = [
            str(_domain_entry(feature_section, str(name)).get("label", str(name).replace("_", " ")))
            for name in item.changed_features
        ]
        if len(feature_labels) == 1:
            change_text = f"Изменение значения признака «{feature_labels[0]}»"
        else:
            change_text = f"Изменение значений признаков: {', '.join(feature_labels)}"
        text = f"{change_text} переводит объект в другой класс в проверенном контрфактическом расчёте."
        title = "Изменить ключевые признаки"
    else:
        text = "Проверенное контрфактическое изменение может изменить класс результата."
        title = "Проверить альтернативный сценарий"
    if technical and item:
        text += f" Наблюдаемый эффект: {item.observed_effect}; действие: {item.actionability}."
    claim_refs, evidence_refs = _refs((claim,))
    return ChangeStatement(title, text, claim_refs, evidence_refs)


def _deduplicate(statements: Sequence[HumanStatement]) -> list[HumanStatement]:
    result: list[HumanStatement] = []
    seen: set[str] = set()
    for statement in statements:
        key = re.sub(r"\W+", " ", statement.title.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        result.append(statement)
    return result


def _validate_domain_user(explanation: HumanExplanation) -> None:
    for term in TECHNICAL_TERMS:
        match = term.search(explanation.user_text)
        if match:
            raise ValueError(f"technical term leaked into domain-user explanation: {match.group(0)}")
    for fragment in explanation.fragments:
        if not fragment.claim_refs or not fragment.evidence_refs:
            raise ValueError("every human fragment must be traceable to claims and evidence")


def compose_human_explanation(
    claims: Sequence[ExplanationClaim],
    graph: ExplanationGraph,
    *,
    action: str,
    audience: str = "domain_user",
    language: str = "ru",
    evidence: ExplanationEvidence | None = None,
    domain_language: Mapping[str, Any] | None = None,
    level: str | None = None,
) -> HumanExplanation:
    """Answer operational user questions without exposing the claim machinery."""

    profile = audience_profile(level or audience)
    domain = dict(domain_language or {})
    evidence = evidence or ExplanationEvidence()
    ranked = rank_human_claims(claims, domain_language=domain)
    technical = profile.show_technical_identifiers

    prediction = next((claim for claim in claims if claim.claim_type == "prediction"), None)
    action_claim = next((claim for claim in claims if claim.claim_type == "recommended_action"), None)
    if prediction is None or action_claim is None:
        raise ValueError("human explanation requires prediction and recommended-action claims")
    decision = _decision_statement(prediction, domain, technical)

    supports: list[ReasonStatement] = []
    contradicts: list[ConcernStatement] = []
    similar_details: list[HumanStatement] = []
    for claim in ranked:
        if claim.claim_type == "feature_contribution":
            reason = _feature_reason(claim, evidence, domain, technical)
            if claim.effect == "favorable":
                supports.append(reason)
            else:
                contradicts.append(ConcernStatement(reason.title, reason.explanation, reason.claim_refs, reason.evidence_refs))
        elif claim.claim_type == "model_rule" and claim.effect != "adverse":
            supports.append(_rule_reason(claim, evidence, technical))
        elif claim.claim_type == "class_concept":
            supports.append(_concept_reason(claim, evidence, domain, technical))
        elif claim.claim_type == "similar_case":
            similar = _similar_statement(claim, evidence, technical)
            similar_details.append(similar)
            if claim.effect == "adverse":
                contradicts.append(ConcernStatement(similar.title, similar.explanation, similar.claim_refs, similar.evidence_refs))
            else:
                supports.append(ReasonStatement(similar.title, similar.explanation, similar.claim_refs, similar.evidence_refs))

    if not supports:
        claim_refs, evidence_refs = _refs((prediction,))
        supports.append(
            ReasonStatement(
                "Совокупность доступных признаков",
                "Доступные признаки в совокупности поддерживают выбранный моделью результат.",
                claim_refs,
                evidence_refs,
            )
        )
    supports = cast(list[ReasonStatement], _deduplicate(supports))

    concern_claims = [claim for claim in ranked if claim.claim_type in {"forgetting", "subgroup_averaging", "lost_rules"}]
    concerns: list[ConcernStatement] = []
    training_details: list[HumanStatement] = []
    if concern_claims:
        training_concern = _training_concern(concern_claims, evidence, technical)
        concerns.append(training_concern)
        training_details.append(training_concern)
    for claim in ranked:
        if claim.claim_type in {"data_deviation", "diagnostic", "missing_channel"}:
            concerns.append(_generic_concern(claim, technical))
    concerns.extend(contradicts)
    concerns = cast(list[ConcernStatement], _deduplicate(concerns))

    action_entry = _domain_entry(_domain_section(domain, "actions"), action)
    default_action = _ACTION_TEXT.get(action, ("Проверить результат", "Перед применением результата требуется дополнительная проверка."))
    action_title = str(action_entry.get("label", default_action[0]))
    action_explanation = str(action_entry.get("explanation", default_action[1]))
    action_refs, action_evidence = _refs((action_claim,))
    action_statement = ActionStatement(action_title, action_explanation, action_refs, action_evidence, action)

    reliability_claims = [action_claim, *[claim for claim in ranked if claim.claim_type in {"diagnostic", "data_quality", "forgetting"}][:2]]
    reliability_refs, reliability_evidence = _refs(reliability_claims)
    if action == "accept":
        reliability_text = "Большинство доступных проверок поддерживает результат; сохраняется обычное ограничение применимости модели."
        reliability_title = "Достаточно для контролируемого применения"
    elif action == "block":
        reliability_text = "Обнаружено критическое противоречие, поэтому результат нельзя применять автоматически."
        reliability_title = "Недостаточно для применения"
    elif action == "insufficient_evidence":
        reliability_text = "Доступных подтверждений недостаточно, чтобы оценить результат как надёжный."
        reliability_title = "Недостаточно подтверждений"
    else:
        reliability_text = "Результат подтверждается частью доступных сведений, но недостаточно надёжен для автоматического решения."
        reliability_title = "Требуется дополнительная проверка"
    reliability = ReliabilityStatement(reliability_title, reliability_text, reliability_refs, reliability_evidence)

    changes = [_change_statement(claim, evidence, domain, technical) for claim in ranked if claim.claim_type == "counterfactual"]
    changes = cast(list[ChangeStatement], _deduplicate(changes))[: profile.max_changes]
    main_reasons = supports[: profile.max_reasons]
    visible_concerns = concerns[: profile.max_concerns]

    technical_metrics: list[HumanStatement] = []
    for claim in ranked:
        if claim.metric_value is None:
            continue
        claim_refs, evidence_refs = _refs((claim,))
        baseline = f"; сравнение: {claim.comparison_baseline}" if claim.comparison_baseline else ""
        technical_metrics.append(
            HumanStatement(
                claim.metric_name or "Технический показатель",
                f"{claim.metric_value:.3f}{baseline}",
                claim_refs,
                evidence_refs,
            )
        )

    explanation = HumanExplanation(
        audience=profile.name,
        language=language,
        decision=decision,
        main_reasons=tuple(main_reasons),
        concerns=tuple(visible_concerns),
        reliability=reliability,
        recommended_action=action_statement,
        what_would_change_result=tuple(changes),
        details=ExplanationDetails(
            supports=tuple(supports),
            contradicts=tuple(contradicts),
            limitations=tuple(concerns),
            training=tuple(training_details),
            similar_cases=tuple(similar_details),
            technical_metrics=tuple(technical_metrics),
        ),
        technical_trace=graph,
    )
    if profile.name == "domain_user":
        _validate_domain_user(explanation)
    return explanation


def explanation_to_text(explanation: HumanExplanation, *, detail: str = "short") -> str:
    """Render user cards first and technical evidence only on explicit request."""

    if detail not in {"short", "full"}:
        raise ValueError("detail must be short or full")
    text = explanation.user_text
    if detail == "short":
        return text
    sections = [
        ("Поддерживает решение", explanation.details.supports),
        ("Противоречит решению", explanation.details.contradicts),
        ("Ограничивает доверие", explanation.details.limitations),
        ("Как модель обучалась", explanation.details.training),
        ("Похожие случаи", explanation.details.similar_cases),
        ("Технические доказательства", explanation.details.technical_metrics),
    ]
    lines = [text.rstrip(), ""]
    for title, statements in sections:
        if not statements:
            continue
        lines.append(f"## {title}")
        lines.extend(f"- **{statement.title}:** {statement.explanation}" for statement in statements)
        lines.append("")
    return "\n".join(lines).strip() + "\n"
