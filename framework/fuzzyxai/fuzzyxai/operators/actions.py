from __future__ import annotations


def select_action(risk: dict, diagnostics: list[dict]) -> dict:
    if int(risk.get("chi_crit", 0)) == 1:
        return {
            "action": "block",
            "status": "blocked",
            "reason_ru": "автоматическое принятие запрещено при chi_crit = 1",
        }
    if diagnostics:
        return {"action": diagnostics[0].get("recommended_action", "audit"), "status": "warning", "reason_ru": "есть диагностическое состояние"}
    return {"action": "accept", "status": "passed", "reason_ru": "критических диагностик нет"}
