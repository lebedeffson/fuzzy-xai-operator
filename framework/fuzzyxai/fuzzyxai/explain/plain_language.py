from __future__ import annotations


ACTION_RU = {
    "accept": "принять",
    "lower_confidence": "понизить доверие",
    "audit": "направить на аудит",
    "defer_to_human": "передать человеку",
    "block": "заблокировать",
    "audit_report": "подготовить аудит-отчёт",
}

COMPONENT_RU = {
    "gamma": "неуверенность или рассогласование",
    "delta": "потеря объяснения",
    "uncertainty": "неуверенность модели",
    "uncertainty_component": "неуверенность модели",
    "reduction": "потеря объяснения",
    "reduction_component": "потеря объяснения",
    "quality": "качество входа",
    "quality_component": "качество входа",
    "conflict": "конфликт источников",
    "conflict_component": "конфликт источников",
    "interval": "интервальная неопределённость",
    "interval_component": "интервальная неопределённость",
}


def action_title_ru(action: str) -> str:
    return ACTION_RU.get(str(action), str(action))


def component_title_ru(component: str) -> str:
    return COMPONENT_RU.get(str(component), str(component))


def explain_action_ru(action: str, rho: float, thresholds: dict | None = None) -> str:
    thresholds = thresholds or {"accept": 0.35, "audit": 0.60, "critical": 0.75}
    accept = float(thresholds.get("accept", 0.35))
    audit = float(thresholds.get("audit", 0.60))
    critical = float(thresholds.get("critical", 0.75))
    if action == "accept":
        return f"ρ = {rho:.2f} ниже порога полного принятия {accept:.2f}. Результат можно использовать в обычном режиме."
    if action == "lower_confidence":
        return (
            f"ρ = {rho:.2f} выше порога полного принятия {accept:.2f}, но ниже порога аудита {audit:.2f}. "
            "Поэтому результат не отклоняется, но доверие к нему ограничивается."
        )
    if action == "audit":
        return f"ρ = {rho:.2f} находится в зоне аудита. Автоматическое принятие недостаточно надёжно: нужна дополнительная проверка."
    if action == "defer_to_human":
        return "Автоматическое решение недостаточно надёжно. Маршрут рекомендует передать результат человеку."
    if action == "block":
        return f"ρ = {rho:.2f} достиг критической зоны {critical:.2f} или активирован критический флаг. Результат нельзя использовать без остановки маршрута."
    return f"Выбрано действие «{action_title_ru(action)}» при ρ = {rho:.2f}."


def explain_gamma_ru(gamma: float) -> str:
    if gamma < 0.15:
        return f"γ = {gamma:.2f}: неуверенность мала, модель выглядит устойчиво для данного маршрута."
    if gamma < 0.45:
        return f"γ = {gamma:.2f}: есть умеренная неуверенность. Результат допустим, но не полностью устойчив."
    return f"γ = {gamma:.2f}: неуверенность высокая и заметно ограничивает доверие к объяснению."


def explain_delta_ru(delta: float) -> str:
    if delta < 0.15:
        return f"Δ = {delta:.2f}: потеря объяснения мала, покрытие объяснения достаточно полное."
    if delta < 0.45:
        return f"Δ = {delta:.2f}: часть объяснительной информации потеряна при редукции."
    return f"Δ = {delta:.2f}: потеря объяснения высокая; пользователю показана только ограниченная часть объяснения."


def explain_representation_ru(representation: str, context: dict | None = None) -> str:
    representation = str(representation or "F0")
    if representation == "F0":
        return "Выбран F0: обычное нечёткое представление для задачи без интервального прогноза, конфликта источников или многоуровневого сигнала."
    if representation == "F_int":
        return "Выбран F_int: результат содержит интервальную неопределённость, поэтому нужно учитывать диапазон возможных состояний."
    if representation == "NAS":
        return "Выбран NAS: обнаружено ограничение источника или конфликт, поэтому риск нельзя объяснять только неуверенностью модели."
    if representation == "F_ML":
        return "Выбран F_ML: объяснение имеет несколько уровней — качество входа, признаки, уверенность модели и итоговое решение."
    return f"Выбран класс представления {representation}."


def explain_main_reason_ru(dominant_component: str) -> str:
    title = component_title_ru(dominant_component)
    if dominant_component in {"delta", "reduction", "reduction_component"}:
        return (
            "Главная причина риска — потеря объяснения. После отбора наиболее важных элементов часть информации "
            "не попала в итоговое объяснение."
        )
    if dominant_component in {"gamma", "uncertainty", "uncertainty_component"}:
        return "Главная причина риска — неуверенность модели или рассогласование с устойчивым объяснением."
    if dominant_component in {"quality", "quality_component"}:
        return "Главная причина риска — ограничение качества входа или источника данных."
    if dominant_component in {"conflict", "conflict_component"}:
        return "Главная причина риска — конфликт источников или объяснительных свидетельств."
    if dominant_component in {"interval", "interval_component"}:
        return "Главная причина риска — интервальная неопределённость результата."
    return f"Главная причина риска — {title}."
