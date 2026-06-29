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


def build_route_from_proof(proof: dict[str, Any], *, proof_ref: str = "") -> OperatorRouteState:
    """Build an operator route from an existing FuzzyXAIProofPackage.

    The builder does not recompute gamma, Delta, rho, or the action. It exposes
    the verified trace as a route state that can be rendered by figures or a site.
    """

    scenario_id = str(proof.get("scenario_id", "unknown"))
    input_values = proof.get("input", {}) or {}
    result = proof.get("computed_result", {}) or {}
    diagnostics = proof.get("diagnostics", []) or []
    diagnostic_id = result.get("diagnostic_id") or (diagnostics[0].get("diagnostic_id") if diagnostics else "")
    operators = _operator_map(proof)
    alignment = _computed(operators, "alignment")
    reduction = _computed(operators, "reduction")
    risk = _computed(operators, "risk_observer")
    action = _computed(operators, "action")
    final_action = str(proof.get("final_action") or result.get("action") or action.get("action", ""))

    nodes = [
        OperatorNodeState(
            node_id="input_artifact",
            title="Входной артефакт",
            input_summary="изображение радужки и признаки качества",
            output_summary="Q_img, Q_seg, p_match",
            value=f"Q_img={_fmt(input_values.get('image_quality'))}; Q_seg={_fmt(input_values.get('segmentation_quality'))}; p={_fmt(input_values.get('model_match_signal'))}",
            status="warning" if float(input_values.get("segmentation_quality", 1.0)) < 0.5 else "passed",
            explanation="Контрольный вход передаёт качество источника и модельный сигнал.",
            formula_ref="адаптер входного артефакта",
            trace_ref="input",
            raw=input_values,
        ),
        OperatorNodeState(
            node_id="adapter",
            title="Адаптер",
            input_summary="сырой артефакт и выход модели",
            output_summary="структурированный вход маршрута",
            value=f"alpha_accept={_fmt(input_values.get('alpha_accept'))}; alpha_block={_fmt(input_values.get('alpha_block'))}",
            status=operators.get("adapter", {}).get("status", "passed"),
            explanation="Адаптер переводит вход и модельный сигнал в форму, пригодную для операторов FuzzyXAI.",
            formula_ref="раздел адаптеров FuzzyXAI",
            trace_ref="operator_values.adapter",
            raw=operators.get("adapter", {}),
        ),
        OperatorNodeState(
            node_id="explanation_object",
            title="Объяснительный объект E_k",
            input_summary="структурированный вход",
            output_summary="E_k с качеством источника и правилом действия",
            value="E_k сформирован",
            status="passed",
            explanation="Объект объяснения связывает p_match, качество источника, правила и trace.",
            formula_ref="E_k",
            trace_ref="computed_result",
            raw=result,
        ),
        OperatorNodeState(
            node_id="uncertainty_class",
            title="Выбор класса F",
            input_summary="E_k и диагностический профиль",
            output_summary="расширенное представление неопределённости",
            value="не F0; требуется учёт качества источника",
            status="warning",
            explanation="Конфликт качества источника не должен редуцироваться до простого нечёткого значения без потерь.",
            formula_ref="выбор F",
            trace_ref="diagnostics",
            raw={"diagnostics": diagnostics},
        ),
        OperatorNodeState(
            node_id="alignment",
            title="Согласование T_ij",
            input_summary="компоненты объяснения",
            output_summary="мера рассогласования gamma",
            value=f"gamma={_fmt(alignment.get('gamma_ij', result.get('gamma')), 3)}",
            threshold=f"gamma_max={_fmt(alignment.get('gamma_max'))}",
            status=operators.get("alignment", {}).get("status", "warning"),
            explanation="T_ij фиксирует рассогласование между высоким модельным сигналом и низким качеством источника.",
            formula_ref="T_ij, gamma",
            trace_ref="operator_values.alignment",
            raw=operators.get("alignment", {}),
        ),
        OperatorNodeState(
            node_id="reduction",
            title="Потери представления",
            input_summary="выбранное представление F",
            output_summary="Delta и r_Delta",
            value=f"Delta={_fmt(reduction.get('delta', result.get('delta')))}; r_Delta={_fmt(reduction.get('r_delta', result.get('r_delta')), 4)}",
            threshold=f"Delta_max={_fmt(reduction.get('delta_max'))}",
            status=operators.get("reduction", {}).get("status", "passed"),
            explanation="Редукция контролируется: потеря представления сохраняется в доказательном следе.",
            formula_ref="Delta, r_Delta",
            trace_ref="operator_values.reduction",
            raw=operators.get("reduction", {}),
        ),
        OperatorNodeState(
            node_id="risk",
            title="Риск rho",
            input_summary="gamma, Delta, критический конфликт",
            output_summary="интегральный риск",
            value=f"rho={_fmt(risk.get('rho', result.get('rho')), 3)}",
            status=operators.get("risk_observer", {}).get("status", "blocked"),
            explanation="Риск-наблюдатель объединяет модельный сигнал, правило блокировки, конфликт источника и r_Delta.",
            formula_ref="rho",
            trace_ref="operator_values.risk_observer",
            raw=operators.get("risk_observer", {}),
        ),
        OperatorNodeState(
            node_id="critical_indicator",
            title="Критический индикатор",
            input_summary="диагностика и риск",
            output_summary="chi_R^crit",
            value=f"chi_R^crit={_fmt(risk.get('chi_R_crit', result.get('chi_R_crit', 1)))}",
            status="blocked" if int(risk.get("chi_R_crit", 0)) == 1 else "passed",
            explanation="Критическая область активна, поэтому принятие результата запрещено.",
            formula_ref="chi_R^crit",
            trace_ref="operator_values.risk_observer",
            raw=risk,
        ),
        OperatorNodeState(
            node_id="diagnostics",
            title="Диагностика D",
            input_summary="операторные статусы",
            output_summary="тип диагностического состояния",
            value=DIAGNOSTIC_RU.get(str(diagnostic_id), str(diagnostic_id)),
            status="blocked" if diagnostic_id else "passed",
            explanation="Диагностика объясняет, почему высокий p_match не приводит к принятию.",
            formula_ref="D_k, D_ij",
            trace_ref="diagnostics",
            raw={"diagnostics": diagnostics},
        ),
        OperatorNodeState(
            node_id="action",
            title="Действие",
            input_summary="rho, chi_R^crit, D",
            output_summary="режим применения",
            value=ACTION_RU.get(final_action, final_action),
            status=operators.get("action", {}).get("status", "blocked"),
            explanation="Для HYBRID-XIRIS итоговое действие — блокировка.",
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
    ]
    return OperatorRouteState(
        scenario_id=scenario_id,
        title=f"FuzzyXAI operator route: {scenario_id}",
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

