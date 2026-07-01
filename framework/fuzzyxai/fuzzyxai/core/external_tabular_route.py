from __future__ import annotations

from fuzzyxai.core.git_info import get_source_commit
from fuzzyxai.core.types import AdaptedInput, OperatorNode, OperatorRoute


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


def build_external_wine_classification_route(adapted: AdaptedInput) -> OperatorRoute:
    values = adapted.values
    probability = float(values["class_probability"])
    missing_rate = float(values.get("missing_rate", 0.0))
    range_violation = float(values.get("feature_range_violation", 0.0))
    uncertainty = round(1.0 - probability, 6)
    quality_penalty = round(max(missing_rate, range_violation), 6)
    gamma = round(max(uncertainty, quality_penalty), 6)
    importance_sum = sum(float(v) for v in values["feature_importance"].values())
    delta = round(max(0.0, 1.0 - min(1.0, importance_sum)), 6)
    rho = round(max(gamma, delta), 6)

    if rho < 0.35:
        action = "accept"
        diagnostic = {
            "diagnostic_id": "D_external_tabular_ok",
            "diagnostic_type": "external_tabular_ok",
            "source": "external_tabular_adapter",
            "criticality": "low",
            "message_ru": "внешняя табличная классификация допустима",
            "recommended_action": "accept",
        }
        risk_status = "passed"
    elif rho < 0.60:
        action = "lower_confidence"
        diagnostic = {
            "diagnostic_id": "D_external_tabular_uncertainty",
            "diagnostic_type": "external_tabular_uncertainty",
            "source": "external_tabular_adapter",
            "criticality": "medium",
            "message_ru": "ограниченная уверенность внешней табличной модели",
            "recommended_action": "lower_confidence",
        }
        risk_status = "warning"
    else:
        action = "audit"
        diagnostic = {
            "diagnostic_id": "D_external_tabular_uncertainty",
            "diagnostic_type": "external_tabular_uncertainty",
            "source": "external_tabular_adapter",
            "criticality": "high",
            "message_ru": "внешняя табличная модель требует аудита",
            "recommended_action": "audit",
        }
        risk_status = "warning"

    computed = {
        "class_probability": probability,
        "uncertainty": uncertainty,
        "quality_penalty": quality_penalty,
        "gamma": gamma,
        "delta": delta,
        "rho": rho,
        "diagnostic_id": diagnostic["diagnostic_id"],
        "action": action,
    }
    nodes = [
        OperatorNode(
            node_id="input_artifact",
            title="Внешняя табличная модель",
            input_summary="payload sklearn wine classifier",
            output_summary="AdaptedInput",
            value=f"p={probability}; class={values['predicted_class']}",
            status="passed",
            explanation="Адаптер принимает внешний результат табличной модели.",
            formula_ref="adapter",
            trace_ref="adapted_input",
            value_source="external_model_payload",
            raw=values,
        ),
        _node("explanation_object", "Объяснительный объект E_k", "E_k сформирован", "passed", "Внешний payload переведён в объяснительный объект.", {"value_source": "computed"}, "E_k"),
        _node("representation", "Класс представления F", "F_tabular", "passed", "Табличный результат представлен через вероятности, признаки и важности.", {"value_source": "computed"}, "F"),
        _node("alignment", "Согласование T_ij", f"gamma={gamma}", "passed" if gamma < 0.35 else "warning", "gamma=max(1-p, quality_penalty).", {"uncertainty": uncertainty, "quality_penalty": quality_penalty, "gamma": gamma, "value_source": "computed"}, "gamma"),
        _node("reduction", "Потери представления", f"Delta={delta}", "passed" if delta < 0.35 else "warning", "Delta=1-sum(feature_importance) для переданного набора важностей.", {"delta": delta, "importance_sum": importance_sum, "value_source": "computed"}, "Delta"),
        _node("risk", "Риск rho", f"rho={rho}", risk_status, "rho=max(gamma, Delta).", {"rho": rho, "theta_accept": 0.35, "theta_warning": 0.60, "value_source": "computed"}, "rho"),
        _node("diagnostics", "Диагностика D", diagnostic["diagnostic_id"], risk_status, diagnostic["message_ru"], {"diagnostics": [diagnostic], "value_source": "computed"}, "D"),
        _node("action", "Действие", action, "passed" if action == "accept" else "warning", "Действие выбрано политикой FuzzyXAI.", {"action": action, "value_source": "computed"}, "action policy"),
        _node("proof", "Доказательный след", "proof trace готов", "passed", "Маршрут сохраняется как проверяемый доказательный след.", {"value_source": "computed"}, "proof trace"),
    ]
    return OperatorRoute(
        scenario_id=adapted.scenario_id,
        title="External Wine Classification FuzzyXAI OperatorRoute",
        nodes=nodes,
        computed_result=computed,
        diagnostics=[diagnostic],
        final_action=action,
        verifier_status="PASS",
        source_commit=get_source_commit(),
    )
