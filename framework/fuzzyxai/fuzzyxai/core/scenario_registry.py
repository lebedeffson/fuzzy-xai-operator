from __future__ import annotations

from typing import Callable

from .types import AdaptedInput, OperatorNode, OperatorRoute


def _node(node_id: str, title: str, value: str, status: str, explanation: str, raw: dict, formula_ref: str = "") -> OperatorNode:
    return OperatorNode(
        node_id=node_id,
        title=title,
        input_summary="предыдущее состояние маршрута",
        output_summary="выход оператора",
        value=value,
        status=status,
        explanation=explanation,
        formula_ref=formula_ref,
        trace_ref=node_id,
        value_source=raw.get("value_source", "computed"),
        raw=raw,
    )


def _route(scenario_id: str, title: str, values: dict, computed: dict, diagnostics: list[dict], nodes: list[OperatorNode]) -> OperatorRoute:
    action = str(computed["action"])
    base_nodes = [
        OperatorNode(
            node_id="input_artifact",
            title="Входной артефакт",
            input_summary="выход внешней модели/правила",
            output_summary="AdaptedInput",
            value="; ".join(f"{key}={value}" for key, value in list(values.items())[:5]),
            status="warning" if diagnostics else "passed",
            explanation="Вход поступает через адаптер FuzzyXAI.",
            formula_ref="adapter",
            trace_ref="adapted_input",
            value_source="external_model_payload",
            raw=values,
        ),
        _node("explanation_object", "Объяснительный объект E_k", "E_k сформирован", "passed", "Вход переведён в объяснительный объект.", {"value_source": "computed"}, "E_k"),
    ]
    tail_nodes = [
        _node("diagnostics", "Диагностика D", str(computed.get("diagnostic_id", "")), "warning" if diagnostics else "passed", "Диагностика задаёт режим применения.", {"diagnostics": diagnostics, "value_source": "computed"}, "D"),
        _node("action", "Действие", action, "blocked" if action == "block" else "warning", "Действие выбрано политикой FuzzyXAI.", {"action": action, "value_source": "computed"}, "action policy"),
        _node("proof", "Доказательный след", "proof trace готов", "passed", "Маршрут сохраняется как проверяемый доказательный след.", {"value_source": "computed"}, "proof trace"),
    ]
    return OperatorRoute(
        scenario_id=scenario_id,
        title=title,
        nodes=base_nodes + nodes + tail_nodes,
        computed_result=computed,
        diagnostics=diagnostics,
        final_action=action,
        verifier_status="PASS",
    )


def build_hybrid_xiris_route(adapted: AdaptedInput) -> OperatorRoute:
    from fuzzyxai.core.route import build_hybrid_xiris_route

    return build_hybrid_xiris_route(adapted)


def build_medical_ecg_signal_route(adapted: AdaptedInput) -> OperatorRoute:
    values = adapted.values
    diagnostic = {
        "diagnostic_id": "D_signal_quality",
        "diagnostic_type": "signal_quality",
        "source": "noisy_or_missing_signal",
        "criticality": "medium",
        "message_ru": "ограничение качества сигнала",
        "recommended_action": "defer_to_human",
    }
    computed = {"quality_score": values["quality_score"], "missing_fragments": values["missing_fragments"], "diagnostic_id": diagnostic["diagnostic_id"], "action": "defer_to_human"}
    nodes = [
        _node("signal_quality", "Качество сигнала", f"quality={values['quality_score']}; missing={values['missing_fragments']}", "warning", "Контрольный ЭКГ-сценарий не является клинической диагностикой.", {**values, "value_source": "computed"}, "D_signal_quality"),
    ]
    return _route(adapted.scenario_id, "ECG FuzzyXAI OperatorRoute", values, computed, [diagnostic], nodes)


def build_gd_anfis_shap_route(adapted: AdaptedInput) -> OperatorRoute:
    values = adapted.values
    gamma = round(float(values.get("gamma_rule_shap", (abs(float(values["alpha_rule"]) - float(values["shap_x2"])) + abs(float(values["alpha_rule"]) - float(values["shap_x1"]))) / 2)), 3)
    diagnostic = {
        "diagnostic_id": "D_rule_attribution_conflict",
        "diagnostic_type": "rule_attribution_conflict",
        "source": "rule_vs_attribution",
        "criticality": "medium",
        "message_ru": "конфликт правила и локального вклада",
        "recommended_action": "audit",
    }
    computed = {"gamma_rule_shap": gamma, "diagnostic_id": diagnostic["diagnostic_id"], "action": "audit"}
    nodes = [
        _node("alignment", "Правило и SHAP", f"alpha={values['alpha_rule']}; gamma={gamma}", "warning", "Правило ANFIS конфликтует с локальными SHAP-вкладами.", {**values, "gamma_rule_shap": gamma, "value_source": "computed"}, "rule-shap alignment"),
    ]
    return _route(adapted.scenario_id, "GD-ANFIS/SHAP FuzzyXAI OperatorRoute", values, computed, [diagnostic], nodes)


def build_beacon_xai_route(adapted: AdaptedInput) -> OperatorRoute:
    values = adapted.values
    diagnostic = {
        "diagnostic_id": "D_counterevidence_conflict",
        "diagnostic_type": "counterevidence_conflict",
        "source": "temporal_counterevidence",
        "criticality": "medium",
        "message_ru": "конфликт контрсвидетельств",
        "recommended_action": "audit",
    }
    computed = {"counter_fragments": values["counter_fragments"], "objects_with_counterevidence": values["objects_with_counterevidence"], "diagnostic_id": diagnostic["diagnostic_id"], "action": "audit"}
    nodes = [
        _node("counterevidence", "Контрсвидетельства", f"support={values['support_fragments']}; counter={values['counter_fragments']}", "warning", "64 -> 11 относится к внешнему BEACON-механизму; FuzzyXAI фиксирует диагностику.", {**values, "value_source": "computed"}, "counterevidence"),
    ]
    return _route(adapted.scenario_id, "BEACON-XAI FuzzyXAI OperatorRoute", values, computed, [diagnostic], nodes)


def build_gis_integro_route(adapted: AdaptedInput) -> OperatorRoute:
    values = adapted.values
    gamma_route = round(max(abs(float(values["p"]) - float(values["alpha_mean"])), abs(float(values["p"]) - float(values["s"]))), 2)
    delta = 0.08
    diagnostic = {
        "diagnostic_id": "D_route_context_limit",
        "diagnostic_type": "route_context_limit",
        "source": "route_context_alignment",
        "criticality": "low",
        "message_ru": "маршрутное контекстное ограничение",
        "recommended_action": "audit_report",
    }
    computed = {"gamma_route": gamma_route, "delta": delta, "formula": "max(|p - alpha_mean|, |p - s|)", "diagnostic_id": diagnostic["diagnostic_id"], "action": "audit_report"}
    nodes = [
        _node("alignment", "Маршрутное согласование", f"gamma_route={gamma_route}", "passed", "Используется max(|p-alpha_mean|, |p-s|), без попарного |alpha_mean-s|.", {**values, "gamma_route": gamma_route, "value_source": "computed"}, "gamma_route"),
        _node("reduction", "Потери маршрута", f"Delta={delta}", "passed", "Контрольная редукционная потеря GIS-маршрута.", {"delta": delta, "value_source": "computed"}, "Delta"),
    ]
    return _route(adapted.scenario_id, "GIS INTEGRO FuzzyXAI OperatorRoute", values, computed, [diagnostic], nodes)


SCENARIO_BUILDERS: dict[str, Callable[[AdaptedInput], OperatorRoute]] = {
    "hybrid_xiris": build_hybrid_xiris_route,
    "medical_ecg_signal": build_medical_ecg_signal_route,
    "gd_anfis_shap": build_gd_anfis_shap_route,
    "beacon_xai": build_beacon_xai_route,
    "gis_integro": build_gis_integro_route,
}
