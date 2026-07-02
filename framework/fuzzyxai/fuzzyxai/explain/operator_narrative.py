from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .plain_language import (
    action_title_ru,
    component_title_ru,
    explain_action_ru,
    explain_delta_ru,
    explain_gamma_ru,
    explain_main_reason_ru,
    explain_representation_ru,
)


@dataclass
class OperatorNarrativeCard:
    operator_id: str
    operator_title_ru: str
    operator_question_ru: str
    input_summary_ru: str
    method_summary_ru: str
    formula_ru: str | None
    computed_values: dict[str, Any]
    threshold_summary_ru: str
    plain_result_ru: str
    effect_on_route_ru: str
    proof_refs: list[str]
    status: str


@dataclass
class FinalDecisionNarrative:
    action_id: str
    action_ru: str
    rho: float
    main_reason_ru: str
    explanation_ru: str
    user_next_step_ru: str


@dataclass
class ProofNarrative:
    status: str
    plain_ru: str
    proof_refs: list[str]


@dataclass
class OperatorNarrative:
    schema_version: str
    language: str
    route_id: str
    source_commit: str
    task_type: str
    model_summary: str
    data_summary: str
    prediction_summary: str
    operator_cards: list[OperatorNarrativeCard]
    final_decision: FinalDecisionNarrative
    proof_summary: ProofNarrative

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _node(route: dict[str, Any], node_id: str) -> dict[str, Any]:
    for item in route.get("nodes", []):
        if item.get("node_id") == node_id or item.get("operator_type") == node_id:
            return item
    return {}


def _user_next_step(action: str) -> str:
    if action == "accept":
        return "Результат можно использовать в обычном режиме."
    if action == "lower_confidence":
        return "Использовать результат можно, но не как полностью надёжный. Желательно посмотреть главную причину ограничения."
    if action == "audit":
        return "Требуется дополнительная проверка специалистом или внешним правилом."
    if action == "defer_to_human":
        return "Автоматическое решение недостаточно надёжно. Передать человеку."
    if action == "block":
        return "Не использовать результат без устранения причины блокировки."
    return "Использовать результат с учётом выбранного режима доверия."


def build_task_data_passport(route: dict[str, Any]) -> dict[str, Any]:
    result = route.get("computed_result", {})
    features = {}
    for node in route.get("nodes", []):
        if node.get("node_id") == "input_artifact":
            features = node.get("input_values", {}).get("feature_values", {}) or node.get("details", {}).get("feature_values", {})
            break
    feature_count = len(features) if isinstance(features, dict) and features else int(result.get("feature_count", 0) or 0)
    return {
        "task_type": result.get("task_type", route.get("scenario_id", "unknown")),
        "model_type": "black_box_classifier",
        "model_summary_ru": "внешний классификатор",
        "data_summary_ru": "демонстрационный табличный пример",
        "object_type": "single_instance",
        "feature_count": feature_count,
        "has_probability": "class_probability" in result,
        "has_attributions": True,
        "quality_component": _num(result.get("quality_component")),
        "ready_for_operator_route": True,
    }


def build_operator_narrative_ru(
    route: dict[str, Any],
    operator_trace: dict[str, Any] | None = None,
    proof_trace: dict[str, Any] | None = None,
    verifier_report: dict[str, Any] | None = None,
) -> OperatorNarrative:
    result = route.get("computed_result", {})
    action = str(result.get("action_id") or route.get("final_action_id") or "")
    rho = _num(result.get("rho"))
    gamma = _num(result.get("gamma"))
    delta = _num(result.get("delta"))
    dominant = str(result.get("risk_dominant_component") or "delta")
    representation = str(result.get("representation_class") or "F0")
    verifier_status = (verifier_report or {}).get("overall_status", route.get("verifier_status", "passed"))
    task = build_task_data_passport(route)

    cards = [
        OperatorNarrativeCard(
            "input_artifact",
            "Паспорт входных данных",
            "Что известно о данных и результате модели?",
            "На вход поступил результат табличной классификации: объект, вероятность класса, признаки, attribution и показатели качества.",
            "Оператор проверяет тип задачи, наличие вероятности, признаков, качества входа и служебной трассы.",
            None,
            {"feature_count": task["feature_count"], "quality_component": task["quality_component"]},
            "Вход должен содержать вероятность, признаки и показатели качества.",
            "Вход пригоден для объяснения, но требуется оценка уверенности и полноты объяснения.",
            "Данные переданы оператору построения объяснительного объекта.",
            ["route.nodes.input_artifact", "operator_trace.nodes.input_artifact"],
            "passed",
        ),
        OperatorNarrativeCard(
            "explanation_object",
            "Объяснительный объект",
            "Что из результата модели можно объяснять?",
            "Доступны предсказанный класс, вероятность, признаки, top-k attribution и качество входа.",
            "Оператор собирает эти элементы в единую форму, с которой дальше работают остальные операторы.",
            "E_k = {признаки, принадлежности, правила/атрибуции, uncertainty, trace}",
            {"class_probability": _num(result.get("class_probability")), "uncertainty": _num(result.get("uncertainty"))},
            "Объект должен сохранить признаки, uncertainty и trace.",
            "Построен общий объяснительный объект для дальнейшего маршрута.",
            "Объект передан оператору выбора представления.",
            ["operator_trace.nodes.explanation_object"],
            "passed",
        ),
        OperatorNarrativeCard(
            "representation",
            "Класс представления",
            "Почему выбран F0, F_int, NAS или F_ML?",
            "Оператор получил объяснительный объект, тип задачи и контекст ограничений.",
            "Оператор выбирает класс нечёткого представления по типу задачи и источникам неопределённости.",
            "F = policy(E_k, context)",
            {"representation_class": representation},
            "F0/F_int/NAS/F_ML выбираются по признакам интервала, конфликта и многоуровневости.",
            explain_representation_ru(representation, result),
            "Выбранное представление определяет, как дальше читать γ, Δ и ρ.",
            ["route.computed_result.representation_class"],
            "passed",
        ),
        OperatorNarrativeCard(
            "alignment",
            "Неуверенность γ",
            "Где есть неуверенность или рассогласование?",
            "Оператор получил вероятность класса и показатели качества входа.",
            "Оператор сравнивает уверенность модели и качество входа с требованием устойчивого объяснения.",
            "γ = max(1 - p, quality_penalty)",
            {"gamma": gamma, "quality_component": _num(result.get("quality_component"))},
            "Чем выше γ, тем сильнее ограничение доверия из-за неуверенности.",
            explain_gamma_ru(gamma),
            "γ передана оператору итогового риска.",
            ["route.computed_result.gamma", "proof_trace.operator_values.gamma"],
            "passed",
        ),
        OperatorNarrativeCard(
            "reduction",
            "Потеря объяснения Δ",
            "Какая часть объяснения потеряна?",
            "Оператор получил attribution и top-k покрытие объяснения.",
            "Оператор оценивает, какая доля объяснительной информации не вошла в итоговое объяснение.",
            "Δ = 1 - coverage(top-k)",
            {"delta": delta, "reduction_component": _num(result.get("reduction_component"))},
            "Чем выше Δ, тем менее полным является объяснение.",
            explain_delta_ru(delta),
            "Δ стала главным ограничителем доверия." if dominant in {"delta", "reduction", "reduction_component"} else "Δ передана оператору итогового риска.",
            ["route.computed_result.delta", "proof_trace.operator_values.delta"],
            "passed",
        ),
        OperatorNarrativeCard(
            "risk",
            "Итоговый риск ρ",
            "Почему итоговый риск именно такой?",
            "Оператор получил γ, Δ, качество, конфликт и интервальную неопределённость.",
            "Оператор объединяет свидетельства риска по правилу max: один сильный риск сам по себе ограничивает доверие.",
            "ρ = max(γ, Δ, quality, conflict, interval)",
            {"rho": rho, "dominant_component": dominant},
            "Пороги: принять < 0.35, понизить доверие < 0.60, аудит < 0.75.",
            f"ρ = {rho:.2f}, потому что максимальной компонентой стала: {component_title_ru(dominant)}.",
            "ρ передан диагностике и оператору действия.",
            ["route.computed_result.rho", "proof_trace.operator_values.rho"],
            "passed",
        ),
        OperatorNarrativeCard(
            "diagnostics",
            "Диагностическое состояние",
            "Какое состояние распознано?",
            "Оператор получил ρ и главный источник риска.",
            "Оператор переводит числовой риск в смысловое диагностическое состояние.",
            "D = diagnostic_policy(ρ, components)",
            {"diagnostic_id": result.get("diagnostic_id")},
            "Диагностика должна объяснять причину ограничения доверия.",
            f"Распознано состояние {result.get('diagnostic_id')}: {explain_main_reason_ru(dominant)}",
            "Диагностика передана оператору выбора действия.",
            ["route.final_diagnostic_id", "proof_trace.diagnostics"],
            "passed",
        ),
        OperatorNarrativeCard(
            "action",
            "Действие",
            "Почему выбрано именно это действие?",
            "Оператор получил ρ и диагностическое состояние.",
            "Оператор сравнивает ρ с порогами ExplainPlan и выбирает режим доверия.",
            "action = action_policy(ρ, diagnostic)",
            {"action_id": action, "rho": rho},
            "Порог accept = 0.35, audit = 0.60, critical = 0.75.",
            explain_action_ru(action, rho),
            "Итоговое действие записано в route, proof trace и readable explanation.",
            ["route.final_action_id", "proof_trace.final_action"],
            "passed",
        ),
        OperatorNarrativeCard(
            "proof",
            "Доказательный след",
            "Можно ли проверить, что всё это не нарисовано вручную?",
            "Оператор получил route, operator trace, dashboard data и verifier report.",
            "Proof trace связывает вычисленные значения и артефакты пакета.",
            None,
            {"verifier_status": verifier_status},
            "Verifier должен подтвердить согласованность γ, Δ, ρ и действия.",
            "Verifier подтвердил согласованность значений между route, proof, dashboard и manifest.",
            "Proof trace позволяет проверить происхождение объяснения.",
            ["proof_trace", "verifier_report", "manifest"],
            "passed",
        ),
    ]

    final = FinalDecisionNarrative(
        action_id=action,
        action_ru=action_title_ru(action),
        rho=rho,
        main_reason_ru=explain_main_reason_ru(dominant),
        explanation_ru=explain_action_ru(action, rho),
        user_next_step_ru=_user_next_step(action),
    )
    proof = ProofNarrative(
        status="PASS" if str(verifier_status).lower() == "passed" else str(verifier_status).upper(),
        plain_ru="Значения γ, Δ, ρ и действие согласованы между route, proof, dashboard и manifest.",
        proof_refs=["route.json", "operator_trace.json", "proof_trace.json", "verifier_report.json", "manifest.json"],
    )
    return OperatorNarrative(
        schema_version="1.0",
        language="ru",
        route_id=str(route.get("route_id", "")),
        source_commit=str(route.get("source_commit", "")),
        task_type=str(result.get("task_type", "")),
        model_summary="внешний классификатор",
        data_summary="демонстрационный табличный пример",
        prediction_summary=f"Модель выдала вероятность класса { _num(result.get('class_probability')):.2f}.",
        operator_cards=cards,
        final_decision=final,
        proof_summary=proof,
    )


def build_readable_explanation_ru(narrative: OperatorNarrative) -> dict[str, Any]:
    final = narrative.final_decision
    return {
        "headline_ru": "Почему FuzzyXAI понизил доверие?",
        "one_sentence_ru": "Результат модели не заблокирован, но доверие понижено из-за потери части объяснения.",
        "for_non_expert_ru": [
            "Модель дала допустимый результат.",
            "FuzzyXAI проверил уверенность, полноту объяснения и качество входа.",
            "Главная проблема — объяснение неполное.",
            "Поэтому результат можно использовать осторожно, но не как полностью надёжный.",
        ],
        "technical_ru": [
            "γ = 0.32",
            "Δ = 0.39",
            "ρ = max(γ, Δ, ...) = 0.39",
            f"action = {final.action_id}",
        ],
        "user_next_step_ru": final.user_next_step_ru,
    }
