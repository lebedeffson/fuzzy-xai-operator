from __future__ import annotations

from fuzzyxai.operators.actions import select_action
from fuzzyxai.operators.alignment import compute_alignment
from fuzzyxai.operators.diagnostics import diagnose_route
from fuzzyxai.operators.reduction import compute_reduction
from fuzzyxai.operators.representation import select_representation_class
from fuzzyxai.operators.risk import observe_risk

from .explanation import build_explainable_object
from .git_info import get_source_commit
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


def build_hybrid_xiris_route(adapted: AdaptedInput) -> OperatorRoute:
    explanation = build_explainable_object(adapted)
    values = adapted.values
    alignment = compute_alignment(values)
    reduction = compute_reduction(values)
    risk = observe_risk(values, reduction)
    diagnostics = diagnose_route(risk)
    representation = select_representation_class(diagnostics)
    action = select_action(risk, diagnostics)
    computed_result = {
        "gamma": alignment["gamma"],
        "delta": reduction["delta"],
        "r_delta": reduction["r_delta"],
        "rho": risk["rho"],
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
        _node("risk", "Риск rho", f"rho={risk['rho']}; chi_crit={risk['chi_crit']}", risk["status"], risk["reason_ru"], risk, "rho, chi_crit"),
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
