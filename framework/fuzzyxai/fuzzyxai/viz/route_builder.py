from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .operator_state import OperatorNodeState, OperatorRouteState


ACTION_RU = {
    "accept": "принятие",
    "lower_confidence": "снижение доверия",
    "request_more_data": "запрос данных",
    "defer_to_human": "передача эксперту",
    "audit": "аудит",
    "audit_report": "отчётный режим",
    "block": "блокировка",
}

DIAGNOSTIC_RU = {
    "D_quality_source_conflict": "конфликт качества источника",
    "D_signal_quality": "ограничение качества сигнала",
    "D_rule_attribution_conflict": "конфликт правила и вклада",
    "D_counterevidence_conflict": "конфликт контрсвидетельств",
    "D_route_context_limit": "ограничение маршрутного контекста",
}

SCENARIO_RU = {
    "hybrid_xiris": {
        "title": "HYBRID-XIRIS: радужка и блокировка",
        "input": "изображение радужки, качество источника и модельный сигнал",
        "explanation": "Контрольный вход показывает конфликт: модель уверена, но качество источника низкое.",
    },
    "medical_ecg_signal": {
        "title": "ECG: сигнал и передача эксперту",
        "input": "ЭКГ-сигнал, шум, пропуски и качество",
        "explanation": "Сценарий не ставит диагноз: он фиксирует ограничение качества сигнала и передаёт случай эксперту.",
    },
    "gd_anfis_shap": {
        "title": "GD-ANFIS/SHAP: конфликт правила и вклада",
        "input": "табличные признаки, правило ANFIS и SHAP-вклады",
        "explanation": "Маршрут выявляет рассогласование между активным правилом и локальными вкладами.",
    },
    "beacon_xai": {
        "title": "BEACON-XAI: временные контрсвидетельства",
        "input": "временные фрагменты, support/counter evidence и batch-сводка",
        "explanation": "Маршрут фиксирует контрсвидетельства и переводит результат в аудит.",
    },
    "gis_integro": {
        "title": "GIS INTEGRO: геослой и маршрутное согласование",
        "input": "геосигнал p, alpha_mean, s и маршрутное согласование",
        "explanation": "Маршрут использует gamma_route = max(|p - alpha_mean|, |p - s|) и отчётный режим.",
    },
}

NODE_RU = {
    "adapter": ("Адаптер", "структурированный вход маршрута"),
    "alignment": ("Согласование", "мера рассогласования"),
    "reduction": ("Потери представления", "Delta / редукционная потеря"),
    "risk_observer": ("Риск-наблюдатель", "rho / критический статус"),
    "counterevidence": ("Контрсвидетельства", "support/counter evidence"),
    "action": ("Действие", "режим применения"),
}


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
        return text if "." in text else f"{text}.0"
    return str(value)


def _operator_map(proof: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item.get("node_id", ""): item for item in proof.get("operator_values", [])}


def _computed(operators: dict[str, dict[str, Any]], node_id: str) -> dict[str, Any]:
    return operators.get(node_id, {}).get("computed", {}) or {}


def _kv(values: dict[str, Any], *, limit: int = 5) -> str:
    parts = []
    for key, value in values.items():
        parts.append(f"{key}={_fmt(value)}")
        if len(parts) >= limit:
            break
    return "; ".join(parts)


def _diagnostic_id(result: dict[str, Any], diagnostics: list[Any]) -> str:
    if result.get("diagnostic_id"):
        return str(result["diagnostic_id"])
    for item in diagnostics:
        if isinstance(item, dict) and item.get("diagnostic_id"):
            return str(item["diagnostic_id"])
    return ""


def build_route_from_proof(proof: dict[str, Any], *, proof_ref: str = "") -> OperatorRouteState:
    """Build an operator route from an existing FuzzyXAIProofPackage.

    The builder does not recompute gamma, Delta, rho, or the action. It exposes
    the verified trace as a route state that can be rendered by figures or a site.
    """

    scenario_id = str(proof.get("scenario_id", "unknown"))
    scenario = SCENARIO_RU.get(scenario_id, {})
    input_values = proof.get("input", {}) or {}
    result = proof.get("computed_result", {}) or {}
    diagnostics = proof.get("diagnostics", []) or []
    diagnostic_id = _diagnostic_id(result, diagnostics)
    operators = _operator_map(proof)
    action = _computed(operators, "action")
    final_action = str(proof.get("final_action") or result.get("action") or action.get("action", ""))

    nodes = [
        OperatorNodeState(
            node_id="input_artifact",
            title="Входной артефакт",
            input_summary=scenario.get("input", "данные сценария"),
            output_summary="параметры для операторного маршрута",
            value=_kv(input_values, limit=5),
            status="warning" if diagnostics else "passed",
            explanation=scenario.get("explanation", "Вход переводится в проверяемое состояние маршрута."),
            formula_ref="адаптер входного артефакта",
            trace_ref="input",
            raw=input_values,
        ),
        OperatorNodeState(
            node_id="explanation_object",
            title="Объяснительный объект E_k",
            input_summary="структурированный вход",
            output_summary="E_k с качеством источника и правилом действия",
            value="E_k сформирован из входа и модели/правила",
            status="passed",
            explanation="Объект объяснения связывает входные признаки, модельный/правиловый сигнал и trace.",
            formula_ref="E_k",
            trace_ref="computed_result",
            raw=result,
        ),
    ]
    for item in proof.get("operator_values", []) or []:
        node_id = str(item.get("node_id", "operator"))
        if node_id == "action":
            continue
        title, output = NODE_RU.get(node_id, (str(item.get("operator_id") or node_id), "выход оператора"))
        computed = item.get("computed", {}) or {}
        nodes.append(
            OperatorNodeState(
                node_id=node_id,
                title=title,
                input_summary="предыдущее состояние маршрута",
                output_summary=output,
                value=_kv(computed, limit=4),
                threshold="",
                status=str(item.get("status", "info")),
                explanation="Вычислительный узел маршрута FuzzyXAI; значение взято из proof package.",
                formula_ref=str(item.get("operator_id", node_id)),
                trace_ref=f"operator_values.{node_id}",
                raw=item,
            )
        )
    nodes.extend([
        OperatorNodeState(
            node_id="diagnostics",
            title="Диагностика D",
            input_summary="операторные статусы",
            output_summary="тип диагностического состояния",
            value=DIAGNOSTIC_RU.get(str(diagnostic_id), str(diagnostic_id)),
            status="warning" if diagnostic_id else "passed",
            explanation="Диагностика объясняет, почему выбран именно этот режим действия.",
            formula_ref="D_k, D_ij",
            trace_ref="diagnostics",
            raw={"diagnostics": diagnostics},
        ),
        OperatorNodeState(
            node_id="action",
            title="Действие",
            input_summary="диагностика и операторные значения",
            output_summary="режим применения",
            value=ACTION_RU.get(final_action, final_action),
            status=str(operators.get("action", {}).get("status", "warning" if final_action != "accept" else "passed")),
            explanation=f"Итоговое действие сценария: {ACTION_RU.get(final_action, final_action)}.",
            formula_ref="политика действия",
            trace_ref="operator_values.action",
            raw=operators.get("action", {}),
        ),
        OperatorNodeState(
            node_id="proof",
            title="Доказательный след",
            input_summary="все operator_values",
            output_summary="проверяемый пакет",
            value=f"verifier={proof.get('verifier_status', 'UNKNOWN')}",
            status="passed" if proof.get("verifier_status") == "PASS" else "warning",
            explanation="След сохраняет computed_result и operator_values для повторной проверки.",
            formula_ref="proof trace",
            trace_ref=proof_ref or "proof package",
            raw={"package_hash": proof.get("package_hash"), "source_commit": proof.get("source_commit")},
        ),
    ])
    return OperatorRouteState(
        scenario_id=scenario_id,
        title=scenario.get("title", f"FuzzyXAI operator route: {scenario_id}"),
        nodes=nodes,
        proof_ref=proof_ref,
        verifier_status=str(proof.get("verifier_status", "")),
        final_action=final_action,
        source_commit=str(proof.get("source_commit", "")),
    )


def load_route_from_proof(path: str | Path) -> OperatorRouteState:
    path = Path(path)
    proof = json.loads(path.read_text(encoding="utf-8"))
    return build_route_from_proof(proof, proof_ref=path.as_posix())
