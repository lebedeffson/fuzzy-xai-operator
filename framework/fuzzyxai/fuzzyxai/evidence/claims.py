from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import EffectDirection, ExplanationClaim, ExplanationEvidence, ExplanationLevel, Severity


def build_explanation_claims(
    evidence: ExplanationEvidence,
    *,
    prediction: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
    action: str,
) -> list[ExplanationClaim]:
    """Translate measured evidence into deterministic, reference-backed claims."""

    claims: list[ExplanationClaim] = []

    def add(
        claim_type: str,
        scope: str,
        subject_id: str,
        statement: str,
        short_statement: str,
        refs: Sequence[str],
        *,
        status: str = "supported",
        effect: EffectDirection = "neutral",
        severity: Severity = "info",
        strength: float | None = None,
        limitations: Sequence[str] = (),
        applicability: str | None = None,
        metric_name: str | None = None,
        metric_value: float | None = None,
        metric_unit: str | None = None,
        comparison_baseline: str | None = None,
        counter_refs: Sequence[str] = (),
        native: bool | None = None,
        surrogate: bool | None = None,
    ) -> None:
        claims.append(
            ExplanationClaim(
                claim_id=f"C-{len(claims) + 1:03d}",
                claim_type=claim_type,
                scope=scope,
                subject_id=subject_id,
                statement=statement,
                short_statement=short_statement,
                evidence_status=status,
                effect=effect,
                severity=severity,
                strength=strength,
                evidence_refs=tuple(refs),
                counter_evidence_refs=tuple(counter_refs),
                limitations=tuple(limitations),
                applicability=applicability,
                metric_name=metric_name,
                metric_value=metric_value,
                metric_unit=metric_unit,
                comparison_baseline=comparison_baseline,
                native=native,
                surrogate=surrogate,
            )
        )

    predictions = prediction.get("predictions")
    predicted_value = predictions[0] if isinstance(predictions, (list, tuple)) and predictions else predictions
    score = prediction.get("score")
    score_text = f" с максимальным модельным баллом {score:.3f}" if isinstance(score, (int, float)) else ""
    add(
        "prediction",
        "object",
        str(predictions),
        f"Модель сформировала прогноз {predictions}{score_text}.",
        f"Прогноз: {predictions}",
        ["prediction"],
        strength=float(score) if isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0 else None,
        metric_name="model_score" if isinstance(score, (int, float)) else None,
        metric_value=float(score) if isinstance(score, (int, float)) else None,
        effect="neutral",
    )

    contribution_method = str(prediction.get("contribution_method", "unknown"))
    contributions = prediction.get("contributions", {})
    # Names already covered by a typed image_region claim are excluded here
    # — otherwise the same measured value (e.g. a region's aggregated
    # contribution passed in under its region name) would surface twice,
    # once as a generic feature_contribution and once as its proper
    # image_region claim, reading as two separate pieces of evidence.
    image_region_names = {region.name for image in evidence.image_representations for region in image.regions}
    if isinstance(contributions, Mapping):
        is_surrogate = "surrogate" in contribution_method.lower()
        # `strength` must be a comparable [0, 1] importance signal, not the
        # raw contribution clamped at 1.0 — raw linear-term contributions
        # routinely exceed 1.0 in magnitude (they're coefficient * feature
        # value, unbounded), so `min(1.0, abs(value))` was clamping nearly
        # every feature to the same strength=1.0 and made downstream ranking
        # (rank_human_claims) unable to tell strong and weak features apart.
        # Normalize relative to the strongest contribution for this object
        # instead: the top feature gets strength=1.0, others scale down.
        max_abs_contribution = max((abs(float(v)) for k, v in contributions.items() if str(k) not in image_region_names), default=0.0)
        for feature, raw_value in sorted(contributions.items(), key=lambda item: abs(float(item[1])), reverse=True):
            if str(feature) in image_region_names:
                continue
            value = float(raw_value)
            relative_strength = abs(value) / max_abs_contribution if max_abs_contribution > 0 else 0.0
            limitations = (
                "Surrogate contribution fidelity is limited to the measured local approximation.",
            ) if is_surrogate else ()
            add(
                "feature_contribution",
                "object",
                str(feature),
                f"Локальный вклад признака {feature} равен {value:+.3f} по методу {contribution_method}.",
                f"Вклад {feature}: {value:+.3f}",
                [f"contribution:{feature}"],
                strength=relative_strength,
                limitations=limitations,
                metric_name="local_contribution",
                metric_value=value,
                comparison_baseline="zero local contribution",
                effect="favorable" if value >= 0 else "adverse",
                native=not is_surrogate,
                surrogate=is_surrogate,
            )

    for item in evidence.data:
        data_ref = f"data:{item.object_id}"
        add(
            "data_quality",
            "object",
            item.object_id,
            f"Качество доступных входных данных объекта {item.object_id} равно {item.data_quality:.3f}.",
            f"Качество данных: {item.data_quality:.3f}",
            [data_ref],
            strength=item.data_quality,
            metric_name="data_quality",
            metric_value=item.data_quality,
            effect="favorable" if item.data_quality >= 0.8 else "adverse",
            severity="info" if item.data_quality >= 0.8 else "warning",
        )
        if item.anomaly_labels:
            features = ", ".join(item.anomaly_labels)
            refs = [f"anomaly:{item.object_id}:{feature}" for feature in item.anomaly_labels]
            add(
                "data_deviation",
                "object",
                item.object_id,
                f"Объект {item.object_id} отклоняется от медианного профиля по признакам {features}.",
                f"Отклонения: {features}",
                refs,
                limitations=("Отклонение само по себе не доказывает ошибку данных.",),
                applicability="rare_or_anomalous_input",
                effect="adverse",
                severity="warning",
            )
            add(
                "data_error_status",
                "object",
                item.object_id,
                "Доступный evidence не подтверждает, что обнаруженное отклонение является ошибкой данных.",
                "Отклонение не классифицировано как ошибка",
                refs,
                status="contested",
                limitations=("Для различения ошибки и редкого подтипа требуется предметная проверка.",),
                effect="mixed",
                severity="warning",
            )

    for trace in evidence.training:
        ref = f"training:{trace.object_id}"
        if trace.first_learned_epoch is not None:
            add(
                "first_learned",
                "object",
                trace.object_id,
                f"Объект {trace.object_id} впервые был правильно распознан на эпохе {trace.first_learned_epoch}.",
                f"Первое обучение: эпоха {trace.first_learned_epoch}",
                [ref],
                metric_name="first_learned_epoch",
                metric_value=float(trace.first_learned_epoch),
                metric_unit="epoch",
                effect="favorable",
            )
        if trace.forgetting_events:
            epochs = ", ".join(str(item) for item in trace.forgetting_events)
            add(
                "forgetting",
                "object",
                trace.object_id,
                f"Для объекта {trace.object_id} зафиксированы события забывания на эпохах {epochs}.",
                f"Забывание: эпохи {epochs}",
                [ref],
                strength=1.0 - trace.stability_score,
                metric_name="forgetting_event_count",
                metric_value=float(len(trace.forgetting_events)),
                metric_unit="events",
                effect="adverse",
                severity="warning",
            )

    for subgroup in evidence.subgroups:
        ref = f"subgroup:{subgroup.subgroup_id}"
        if subgroup.averaged:
            add(
                "subgroup_averaging",
                "subgroup",
                subgroup.subgroup_id,
                f"При изменении общей метрики на {subgroup.global_metric_change:+.3f} метрика подгруппы {subgroup.subgroup_id} изменилась на {subgroup.subgroup_metric_change:+.3f}.",
                f"Подгруппа {subgroup.subgroup_id} усредняется",
                [ref],
                limitations=subgroup.limitations,
                metric_name="subgroup_metric_change",
                metric_value=subgroup.subgroup_metric_change,
                comparison_baseline=f"global_metric_change={subgroup.global_metric_change:+.3f}",
                effect="adverse",
                severity="warning",
            )
        if subgroup.disappeared_rules:
            add(
                "lost_rules",
                "subgroup",
                subgroup.subgroup_id,
                f"В истории подгруппы {subgroup.subgroup_id} перестали наблюдаться правила {', '.join(subgroup.disappeared_rules)}.",
                f"Потеряны правила: {', '.join(subgroup.disappeared_rules)}",
                [ref],
                limitations=subgroup.limitations,
                effect="adverse",
                severity="warning",
            )

    for rule in evidence.rules:
        ref = f"rule:{rule.rule_id}"
        provenance = "нативным" if rule.native else "суррогатным"
        limitations = []
        if rule.surrogate:
            limitations.append("Суррогатное правило описывает поведение модели только в пределах измеренной fidelity.")
        add(
            "model_rule",
            "model",
            rule.rule_id,
            f"Правило {rule.rule_id} является {provenance}: {rule.human_text}.",
            f"{rule.rule_id}: {rule.human_text}",
            [ref],
            strength=rule.importance if rule.importance is not None and 0.0 <= rule.importance <= 1.0 else None,
            limitations=limitations,
            metric_name="rule_importance" if rule.importance is not None else None,
            metric_value=rule.importance,
            effect="mixed" if rule.is_conflicting else "favorable",
            severity="warning" if rule.is_conflicting else "info",
            native=rule.native,
            surrogate=rule.surrogate,
        )

    for concept in evidence.concepts:
        ref = f"concept:{concept.class_id}"
        add(
            "class_concept",
            "class",
            concept.class_id,
            concept.human_description,
            f"Концепт класса {concept.class_name}",
            [ref],
            limitations=concept.limitations,
            metric_name="primary_rule_coverage" if concept.primary_rule_coverage is not None else None,
            metric_value=concept.primary_rule_coverage,
            effect="favorable",
        )

    for case in evidence.similar_cases:
        ref = f"similar:{case.query_object_id}:{case.reference_object_id}"
        add(
            "similar_case",
            "object",
            case.query_object_id,
            f"Объект {case.reference_object_id} имеет сходство {case.similarity_score:.3f} с объектом {case.query_object_id}; использован метод {case.similarity_method} для представления «{case.compared_representation}».",
            f"Похожий случай {case.reference_object_id}: {case.similarity_score:.3f}",
            [ref],
            strength=case.similarity_score,
            limitations=case.limitations,
            metric_name="similarity",
            metric_value=case.similarity_score,
            comparison_baseline=case.compared_representation,
            effect=(
                "adverse"
                if case.is_counterexample
                else "favorable"
                if case.reference_label == predicted_value or case.reference_prediction == predicted_value
                else "mixed"
            ),
        )

    for image in evidence.image_representations:
        measured_regions = [region for region in image.regions if region.contribution is not None]
        max_abs_region_contribution = max((abs(float(region.contribution)) for region in measured_regions), default=0.0)  # type: ignore[arg-type]
        for region in image.regions:
            ref = f"image_region:{image.object_id}:{region.name}"
            if region.contribution is None:
                add(
                    "image_region",
                    "object",
                    region.name,
                    f"Область «{region.name}» на изображении {image.object_id} содержит {region.pixel_count} пикселей; вклад в прогноз не измерен для этой области.",
                    f"Область {region.name}: не измерено",
                    [ref],
                    effect="neutral",
                    metric_name="region_pixel_count",
                    metric_value=float(region.pixel_count),
                )
                continue
            relative_strength = abs(float(region.contribution)) / max_abs_region_contribution if max_abs_region_contribution > 0 else 0.0
            add(
                "image_region",
                "object",
                region.name,
                f"Область «{region.name}» на изображении {image.object_id} ({region.pixel_count} пикселей) имеет вклад {region.contribution:+.3f} в прогноз.",
                f"Область {region.name}: {region.contribution:+.3f}",
                [ref],
                strength=relative_strength,
                metric_name="region_contribution",
                metric_value=float(region.contribution),
                effect="favorable" if region.contribution >= 0 else "adverse",
            )

    for activation in evidence.fuzzy_rule_activations:
        ref = f"fuzzy_rule:{activation.object_id}:{activation.rule_id}"
        terms_text = "; ".join(f"{term.feature} соответствует терму «{term.term}» (степень {term.membership_degree:.2f})" for term in activation.terms)
        add(
            "fuzzy_rule",
            "object",
            activation.rule_id,
            f"Правило {activation.rule_id} активировано со степенью {activation.activation_strength:.2f}: {terms_text}. Заключение правила: класс {activation.conclusion}.",
            f"Правило {activation.rule_id}: степень {activation.activation_strength:.2f}",
            [ref],
            strength=activation.activation_strength,
            limitations=activation.limitations,
            metric_name="activation_strength",
            metric_value=activation.activation_strength,
            effect="favorable" if str(activation.conclusion) == str(predicted_value) else "adverse",
        )

    for index, counterfactual in enumerate(evidence.counterfactuals):
        ref = f"counterfactual:{index}"
        changed = dict(counterfactual.changed_features) or {"rules": list(counterfactual.changed_rules)}
        add(
            "counterfactual",
            "object",
            str(counterfactual.source_prediction),
            f"Контрфактическое изменение {changed} переводит прогноз из {counterfactual.source_prediction} в {counterfactual.target_prediction}.",
            f"Контрфакт: {counterfactual.source_prediction} → {counterfactual.target_prediction}",
            [ref],
            strength=counterfactual.stability,
            limitations=counterfactual.limitations,
            applicability=counterfactual.actionability,
            metric_name="observed_effect" if counterfactual.observed_effect is not None else None,
            metric_value=counterfactual.observed_effect,
            effect="mixed",
        )

    diagnostic_refs: list[str] = []
    for index, diagnostic in enumerate(diagnostics):
        ref = f"diagnostic:{index}"
        diagnostic_refs.append(ref)
        reason = str(diagnostic.get("reason", diagnostic.get("code", "diagnostic")))
        status = "insufficient_evidence" if str(diagnostic.get("code", "")).endswith("_missing") else "supported"
        add(
            "diagnostic",
            "model",
            str(diagnostic.get("code", index)),
            f"Диагностика: {reason}.",
            reason,
            [ref],
            status=status,
            limitations=("Автоматическое принятие ограничено до устранения диагностики.",),
            effect="adverse",
            severity="critical" if diagnostic.get("severity") == "critical" else "warning",
        )

    for missing in evidence.missing:
        add(
            "missing_channel",
            "model",
            missing,
            f"Канал evidence «{missing}» недоступен; связанные с ним утверждения не формируются.",
            f"Нет evidence: {missing}",
            ["trace:missing_evidence"],
            status="insufficient_evidence",
            effect="unknown",
            severity="warning",
            limitations=(f"Канал evidence «{missing}» отсутствует.",),
        )

    add(
        "recommended_action",
        "object",
        str(predictions),
        f"На основании доступных claims рекомендовано действие {action}.",
        f"Действие: {action}",
        diagnostic_refs or ["prediction"],
        status="insufficient_evidence" if action == "insufficient_evidence" else "supported",
        applicability=action,
        effect="adverse" if action in {"review", "audit", "block", "insufficient_evidence", "defer_to_human"} else "favorable",
        severity="critical" if action == "block" else "warning" if action != "accept" else "info",
        limitations=("Действие ограничено доступными evidence-каналами.",) if action == "insufficient_evidence" else (),
    )
    return claims


def determine_explanation_level(
    evidence: ExplanationEvidence,
    *,
    contribution_method: str | None,
    operator_channels: Mapping[str, bool],
) -> ExplanationLevel:
    """Determine the highest fully evidenced level without promoting missing channels."""

    available = ["prediction", "call_trace"]
    native = ["prediction", "call_trace"]
    surrogate: list[str] = []
    if evidence.data:
        available.append("data_profile")
        native.append("data_profile")
    if contribution_method:
        available.append("local_contributions")
        (surrogate if "surrogate" in contribution_method else native).append("local_contributions")
    if evidence.rules:
        available.append("rules")
        if any(rule.native for rule in evidence.rules):
            native.append("rules")
        if any(rule.surrogate for rule in evidence.rules):
            surrogate.append("rules")
    if evidence.concepts:
        available.append("class_concepts")
        native.append("class_concepts")
    if evidence.fuzzy_rule_activations:
        available.append("fuzzy_rule_activations")
        native.append("fuzzy_rule_activations")
    if evidence.image_representations:
        available.append("image_representation")
        native.append("image_representation")
    if evidence.similar_cases:
        available.append("similar_cases")
        native.append("similar_cases")
    if evidence.training or evidence.subgroups:
        available.append("training_history")
        native.append("training_history")
    if evidence.counterfactuals:
        available.append("counterfactuals")
        native.append("counterfactuals")
    for channel in ("alignment", "reduction", "risk"):
        if operator_channels.get(channel, False):
            available.append(channel)
            native.append(channel)

    level = "E0"
    if "data_profile" in available:
        level = "E1"
    if any(channel in available for channel in ("local_contributions", "rules", "fuzzy_rule_activations")):
        level = "E2"
    if any(channel in available for channel in ("class_concepts", "similar_cases")):
        level = "E3"
    if "training_history" in available:
        level = "E4"
    full = {"alignment", "reduction", "risk", "counterfactuals"}
    if full <= set(available):
        level = "E5"

    expected = {
        "prediction",
        "call_trace",
        "data_profile",
        "local_contributions",
        "rules",
        "class_concepts",
        "similar_cases",
        "training_history",
        "counterfactuals",
        "alignment",
        "reduction",
        "risk",
    }
    missing = sorted(expected - set(available))
    rationale = {
        "E0": "Доступны только прогноз и trace вызова.",
        "E1": "Доступны профиль входных данных и базовая локальная evidence.",
        "E2": "Доступны внутренние вклады, правила или явно маркированный суррогат.",
        "E3": "Доступны концепты классов или похожие случаи.",
        "E4": "Доступна наблюдаемая история обучения и forgetting evidence.",
        "E5": "Доступен полный операторный маршрут, контрфакты и аудит.",
    }[level]
    return ExplanationLevel(
        level=level,
        available_channels=tuple(dict.fromkeys(available)),
        missing_channels=tuple(missing),
        native_channels=tuple(dict.fromkeys(native)),
        surrogate_channels=tuple(dict.fromkeys(surrogate)),
        rationale=rationale,
    )
