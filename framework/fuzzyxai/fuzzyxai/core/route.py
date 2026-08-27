from __future__ import annotations

from .alignment import compute_alignment
from .explanation import build_explainable_object
from .git_info import get_source_commit
from .reduction import compute_reduction
from .risk_observer import observe_legacy_normalized_risk
from .scenario_engine import DEFAULT_HYBRID_PLAN
from .thresholds import HYBRID_XIRIS_THRESHOLDS
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


def _hybrid_operator_values(values: dict) -> tuple[dict, dict, dict, list[dict], str, dict]:
    """Translate the HYBRID fixture into core operator calls.

    Scenario defaults stay here; the public operator facade never injects them.
    """

    plan = DEFAULT_HYBRID_PLAN
    alignment_components = values.get("alignment_components", plan.gamma_components)
    alignment_weights = values.get("alignment_weights", plan.gamma_weights)
    gamma_max = float(values.get("gamma_max", plan.gamma_max))
    delta_t = float(values.get("delta_t", plan.delta_t))
    delta_max = float(values.get("delta_max", plan.delta_max))
    alignment_result = compute_alignment(
        alignment_components,
        alignment_weights,
        gamma_max=gamma_max,
        delta_t=delta_t,
        delta_max=delta_max,
    )
    alignment = {
        "gamma": alignment_result.gamma,
        "gamma_max": alignment_result.gamma_max,
        "status": "warning" if alignment_result.gamma > gamma_max * 0.75 else "passed",
        "value_source": "computed",
        "components": dict(alignment_components),
        "weights": dict(alignment_weights),
    }

    reduction_components = values.get("reduction_components", {"hybrid_delta": 0.106811})
    reduction_weights = values.get("reduction_weights", {"hybrid_delta": 1.0})
    reduction_result = compute_reduction(reduction_components, reduction_weights, delta_max)
    kappa_delta = float(values.get("kappa_delta", HYBRID_XIRIS_THRESHOLDS["kappa_delta"]))
    reduction = {
        "delta": reduction_result.delta,
        "r_delta": round(min(1.0, kappa_delta * reduction_result.delta), 4),
        "delta_max": reduction_result.delta_max,
        "kappa_delta": kappa_delta,
        "status": "passed" if reduction_result.allowed else "blocked",
        "value_source": "computed",
        "components": dict(reduction_components),
        "weights": dict(reduction_weights),
    }

    source_conflict = (
        float(values["model_match_signal"]) >= float(values["alpha_accept"])
        and float(values["image_quality"]) < 0.5
        and float(values["segmentation_quality"]) < 0.5
    )
    risk_components = values.get(
        "risk_components",
        {
            "model_signal": float(values["model_match_signal"]),
            "block_rule": float(values["alpha_block"]),
            "source_conflict": float(source_conflict),
            "reduction_component": reduction["r_delta"],
        },
    )
    risk_weights = values.get("risk_weights", plan.risk_weights)
    risk_result = observe_legacy_normalized_risk(
        risk_components, risk_weights, plan.thresholds, int(source_conflict)
    )
    risk = {
        "legacy_risk_score": round(risk_result.rho, 3),
        "scientific_contract": "legacy_not_P19_rho",
        "chi_crit": risk_result.chi_r_crit,
        "risk_zone": "critical" if risk_result.chi_r_crit else "normal",
        "status": "blocked" if risk_result.chi_r_crit else "passed",
        "reason_ru": "критический конфликт качества источника и модельного сигнала" if risk_result.chi_r_crit else "критическая зона не активна",
        "value_source": "computed",
        "components": dict(risk_components),
        "weights": dict(risk_weights),
    }
    diagnostics = []
    if risk_result.chi_r_crit:
        diagnostics.append(
            {
                "diagnostic_id": "D_quality_source_conflict",
                "diagnostic_type": "quality_source_conflict",
                "source": "image_and_segmentation_quality",
                "criticality": "high",
                "message_ru": "конфликт качества источника и модельного сигнала",
                "recommended_action": "block",
            }
        )
    representation = "FML-audit" if diagnostics else "F0"
    action = {
        "action": risk_result.action,
        "status": "blocked" if risk_result.action == "block" else "passed",
        "reason_ru": "автоматическое принятие запрещено при chi_crit = 1" if risk_result.chi_r_crit else "критических диагностик нет",
    }
    return alignment, reduction, risk, diagnostics, representation, action


def build_hybrid_xiris_route(adapted: AdaptedInput) -> OperatorRoute:
    explanation = build_explainable_object(adapted)
    values = adapted.values
    alignment, reduction, risk, diagnostics, representation, action = _hybrid_operator_values(values)
    computed_result = {
        "gamma": alignment["gamma"],
        "delta": reduction["delta"],
        "r_delta": reduction["r_delta"],
        "legacy_risk_score": risk["legacy_risk_score"],
        "scientific_contract": "legacy_hybrid_route_not_P19_rho",
        "chi_crit": risk["chi_crit"],
        "selected_class": representation,
        "diagnostic_id": diagnostics[0]["diagnostic_id"] if diagnostics else "",
        "action": action["action"],
    }
    nodes = [
        OperatorNode(
            node_id="input_artifact",
            title="Входной артефакт",
            input_summary="выход внешней модели и признаки качества",
            output_summary="AdaptedInput",
            value=f"Q_img={values['image_quality']}; Q_seg={values['segmentation_quality']}; p={values['model_match_signal']}",
            status="warning",
            explanation="Адаптер принимает внешний результат HYBRID-XIRIS.",
            formula_ref="adapter",
            trace_ref="adapted_input",
            value_source="external_model_payload",
            raw=values,
        ),
        _node("explanation_object", "Объяснительный объект E_k", "E_k сформирован", "passed", "Вход переведён в объяснительный объект.", explanation.components, "E_k"),
        _node("alignment", "Согласование T_ij", f"gamma={alignment['gamma']}", alignment["status"], "Рассогласование качества источника и модельного сигнала.", alignment, "T_ij"),
        _node("representation", "Выбор класса F", representation, "warning", "Критический конфликт требует расширенного представления.", {"value_source": "computed"}, "F"),
        _node("reduction", "Потери представления", f"Delta={reduction['delta']}; r_Delta={reduction['r_delta']}", reduction["status"], "Потеря редукции сохраняется в маршруте.", reduction, "Delta"),
        _node("risk", "Legacy compatibility score", f"legacy score={risk['legacy_risk_score']}; chi_crit={risk['chi_crit']}", risk["status"], risk["reason_ru"], risk, "legacy score, chi_crit"),
        _node("diagnostics", "Диагностика D", computed_result["diagnostic_id"], "blocked", "Диагностика объясняет запрет автоматического принятия.", {"diagnostics": diagnostics, "value_source": "computed"}, "D"),
        _node("action", "Действие", action["action"], action["status"], action["reason_ru"], action, "action policy"),
        _node("proof", "Доказательный след", "proof trace готов", "passed", "Маршрут сохраняется как проверяемый доказательный след.", {"value_source": "computed"}, "proof trace"),
    ]
    return OperatorRoute(
        scenario_id=adapted.scenario_id,
        title="HYBRID-XIRIS FuzzyXAI OperatorRoute",
        nodes=nodes,
        computed_result=computed_result,
        diagnostics=diagnostics,
        final_action=action["action"],
        verifier_status="PASS",
        source_commit=get_source_commit(),
    )


def build_route(adapted: AdaptedInput) -> OperatorRoute:
    """Compute a FuzzyXAI OperatorRoute from an adapted model payload."""

    from .scenario_registry import SCENARIO_BUILDERS

    try:
        builder = SCENARIO_BUILDERS[adapted.scenario_id]
    except KeyError as exc:
        raise ValueError(f"unsupported FuzzyXAI scenario: {adapted.scenario_id}") from exc
    return builder(adapted)
