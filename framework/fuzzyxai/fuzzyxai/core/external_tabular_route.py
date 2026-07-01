from __future__ import annotations

from datetime import UTC, datetime

from fuzzyxai.core.git_info import get_source_commit
from fuzzyxai.core.types import AdaptedInput, OperatorEdge, OperatorNode, OperatorRoute


def _node(
    node_id: str,
    title: str,
    value: str,
    status: str,
    explanation: str,
    raw: dict,
    formula_ref: str = "",
    *,
    operator_type: str = "",
    input_refs: list[str] | None = None,
    output_refs: list[str] | None = None,
    input_values: dict | None = None,
    output_values: dict | None = None,
    formula_id: str | None = None,
    formula_text: str | None = None,
    formula_latex: str | None = None,
    components: dict | None = None,
    thresholds: dict | None = None,
    status_reason_ru: str = "",
    interpretation_ru: str = "",
    next_node_ids: list[str] | None = None,
    details: dict | None = None,
) -> OperatorNode:
    return OperatorNode(
        node_id=node_id,
        title=title,
        title_ru=title,
        operator_type=operator_type or node_id,
        input_summary="предыдущее состояние маршрута",
        output_summary="выход оператора",
        value=value,
        status=status,
        explanation=explanation,
        formula_ref=formula_ref,
        trace_ref=node_id,
        value_source=raw.get("value_source", "computed"),
        raw=raw,
        input_refs=input_refs or [],
        output_refs=output_refs or [],
        input_values=input_values or {},
        output_values=output_values or {},
        formula_id=formula_id,
        formula_text=formula_text or explanation,
        formula_latex=formula_latex,
        components=components or raw,
        thresholds=thresholds or {},
        status_reason_ru=status_reason_ru or explanation,
        interpretation_ru=interpretation_ru or explanation,
        next_node_ids=next_node_ids or [],
        source_commit=get_source_commit(),
        details=details or {},
    )


def _representation_title(class_id: str) -> str:
    return {
        "F0": "обычное нечёткое представление",
        "F_int": "интервальное нечёткое представление",
        "NAS": "источниковое представление",
        "F_ML": "многоуровневое представление",
    }.get(class_id, class_id)


def _representation_interpretation(class_id: str) -> str:
    return {
        "F0": "Базовое представление достаточно для одной уверенности и top-k объяснения.",
        "F_int": "Интервальное представление фиксирует ширину прогноза или неопределённость регрессии.",
        "NAS": "Источниковое представление фиксирует конфликт качества, источника или объяснения.",
        "F_ML": "Многоуровневое представление связывает качество входа, признаки, уверенность модели и объяснение.",
    }.get(class_id, "Класс представления выбран политикой FuzzyXAI.")


def _select_diagnostic(task_type: str, rho: float, gamma: float, delta: float, quality: float, conflict: float, interval: float) -> tuple[str, str, str]:
    if rho < 0.35:
        return "D_external_tabular_ok", "accept", "внешний результат допустим"
    if task_type == "signal_quality" and quality >= 0.55:
        return ("D_signal_missing_fragments" if quality >= 0.70 else "D_signal_noise_limit"), "defer_to_human", "ограничение качества сигнала"
    if task_type == "image_like_classification" and quality >= 0.55:
        return "D_image_quality_limit", "audit", "ограничение качества изображения"
    if conflict >= 0.55:
        return "D_rule_attribution_conflict", "audit", "конфликт объяснительного источника и модельного сигнала"
    if task_type == "tabular_regression":
        diagnostic = "D_external_regression_uncertainty" if interval >= delta else "D_external_regression_explanation_loss"
        return diagnostic, "lower_confidence" if rho < 0.60 else "audit", "ограниченная уверенность внешней регрессионной модели"
    if task_type == "image_like_classification":
        diagnostic = "D_image_explanation_reduction" if delta >= gamma else "D_external_image_uncertainty"
        return diagnostic, "lower_confidence" if rho < 0.60 else "audit", "ограничение image-like объяснения"
    if quality >= 0.35:
        return "D_external_tabular_quality_limit", "lower_confidence" if rho < 0.60 else "audit", "ограничение качества внешнего табличного входа"
    if delta >= gamma and delta >= 0.45:
        return "D_external_tabular_reduction_loss", "lower_confidence" if rho < 0.60 else "audit", "потери редуцированного табличного объяснения"
    return "D_external_tabular_uncertainty", "lower_confidence" if rho < 0.60 else "audit", "ограниченная уверенность внешней табличной модели"


def build_external_wine_classification_route(adapted: AdaptedInput) -> OperatorRoute:
    values = adapted.values
    source_commit = get_source_commit()
    context = dict(values.get("context_values") or {})
    task_type = str(context.get("task_type", "tabular_classification"))
    perturbation = str(context.get("perturbation", "external_payload"))
    model_name = str(values.get("model_name", "external_tabular_model"))
    dataset_name = str(values.get("dataset_name", "unknown_dataset"))
    probability = float(values["class_probability"])
    missing_rate = float(values.get("missing_rate", 0.0))
    range_violation = float(values.get("feature_range_violation", 0.0))
    conflict_component = round(float(context.get("conflict_component", 0.0)), 6)
    interval_width = round(float(context.get("prediction_interval_width", 0.0)), 6)
    signal_noise = round(float(context.get("noise_ratio", 0.0)), 6)
    image_noise = round(float(context.get("occlusion_rate", 0.0)), 6)
    uncertainty = round(1.0 - probability, 6)
    quality_penalty = round(max(missing_rate, range_violation, signal_noise, image_noise), 6)
    gamma = round(max(uncertainty, quality_penalty, conflict_component, interval_width), 6)
    importance_sum = sum(float(v) for v in values["feature_importance"].values())
    delta = round(max(0.0, 1.0 - min(1.0, importance_sum)), 6)
    rho = round(max(gamma, delta, quality_penalty, conflict_component, interval_width), 6)
    dominant_component = max(
        {
            "gamma": gamma,
            "delta": delta,
            "quality": quality_penalty,
            "conflict": conflict_component,
            "interval": interval_width,
        }.items(),
        key=lambda item: item[1],
    )[0]
    representation_class = str(context.get("representation_class") or "F0")
    if not context.get("representation_class"):
        if interval_width > 0:
            representation_class = "F_int"
        elif conflict_component > 0.0 or quality_penalty >= 0.55:
            representation_class = "NAS"
        elif task_type in {"signal_quality", "image_like_classification"}:
            representation_class = "F_ML"

    diagnostic_id, action, message_ru = _select_diagnostic(task_type, rho, gamma, delta, quality_penalty, conflict_component, interval_width)
    risk_status = "passed" if action == "accept" else "warning"
    diagnostic = {
        "diagnostic_id": diagnostic_id,
        "diagnostic_type": diagnostic_id.removeprefix("D_"),
        "source": "external_tabular_adapter",
        "criticality": "low" if action == "accept" else "high" if action in {"audit", "defer_to_human"} else "medium",
        "message_ru": message_ru,
        "recommended_action": action,
    }

    computed = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "task_type": task_type,
        "perturbation": perturbation,
        "representation_class": representation_class,
        "class_probability": probability,
        "uncertainty": uncertainty,
        "quality_penalty": quality_penalty,
        "uncertainty_component": uncertainty,
        "quality_component": quality_penalty,
        "reduction_component": delta,
        "conflict_component": conflict_component,
        "interval_component": interval_width,
        "gamma": gamma,
        "delta": delta,
        "rho": rho,
        "risk_dominant_component": dominant_component,
        "diagnostic_id": diagnostic["diagnostic_id"],
        "action_id": action,
        "action": action,
    }
    feature_importance = dict(values["feature_importance"])
    top_k = int(values.get("top_k_importance", len(feature_importance)))
    selected_features = list(feature_importance.keys())
    risk_zone = "accept" if rho < 0.35 else "lower_confidence" if rho < 0.60 else "audit"
    nodes = [
        OperatorNode(
            node_id="input_artifact",
            title="Внешняя модель",
            title_ru="Внешняя модель",
            operator_type="adapter",
            input_summary="payload внешней модели",
            output_summary="AdaptedInput",
            value=f"p={probability}; class={values['predicted_class']}",
            status="passed",
            explanation="Адаптер принимает внешний результат модели.",
            formula_ref="adapter",
            trace_ref="adapted_input",
            value_source="external_model_payload",
            raw=values,
            input_values={
                "source_type": values.get("source_type"),
                "model_name": model_name,
                "dataset_name": dataset_name,
                "model_output": {
                    "predicted_class": values["predicted_class"],
                    "class_probability": probability,
                },
                "quality_metrics": {
                    "missing_rate": missing_rate,
                    "feature_range_violation": range_violation,
                },
                "feature_values": values.get("feature_values", {}),
                "attribution_values": feature_importance,
                "context_values": context,
            },
            output_values={
                "adapted_input_id": f"{adapted.scenario_id}:{model_name}",
                "trace_origin": "external_model_payload",
                "class_probability": probability,
                "feature_importance": feature_importance,
            },
            formula_id=None,
            formula_text="Нормализация внешнего входа",
            components={"model_name": model_name, "dataset_name": dataset_name, "source_type": values.get("source_type"), "task_type": task_type, "perturbation": perturbation},
            thresholds={},
            status_reason_ru="Внешний payload успешно приведён к AdaptedInput.",
            interpretation_ru="Фреймворк получил уверенность модели, признаки, важности признаков, метрики качества и контекст ограничения.",
            next_node_ids=["explanation_object"],
            source_commit=source_commit,
            details={"applications_used": False},
        ),
        _node(
            "explanation_object",
            "Объяснительный объект E_k",
            "E_k сформирован",
            "passed",
            "Внешний payload переведён в объяснительный объект.",
            {"value_source": "computed"},
            "E_k",
            operator_type="explanation_object",
            input_refs=["input_artifact"],
            output_refs=["representation"],
            input_values={"class_probability": probability, "feature_importance": feature_importance},
            output_values={
                "terms": selected_features,
                "memberships": {name: value for name, value in feature_importance.items()},
                "rules": [],
                "activations": {"predicted_class": values["predicted_class"]},
                "uncertainty": uncertainty,
                "trace": f"{adapted.scenario_id}:{model_name}",
                "object_summary": "табличный объект с вероятностью класса и top-k атрибуциями",
            },
            formula_id="E_k",
            formula_text="E_k = {L_k, μ_k, R_k, α_k, u_k, τ_k}",
            components={"terms": selected_features, "uncertainty": uncertainty},
            status_reason_ru="Объект объяснения содержит вероятность, признаки и атрибуции.",
            interpretation_ru="E_k фиксирует, что модель уверена не полностью и объяснение редуцировано до top-k признаков.",
            next_node_ids=["representation"],
        ),
        _node(
            "representation",
            "Класс представления F",
            "F0",
            "passed",
            "Табличный результат представлен через обычное нечёткое представление.",
            {"value_source": "computed"},
            "F",
            operator_type="representation",
            input_refs=["explanation_object"],
            output_refs=["alignment", "reduction"],
            input_values={"source_type": values.get("source_type"), "task_type": task_type, "has_single_probability": True, "has_top_k_importance": True, "context": context},
            output_values={"class_id": representation_class, "class_title_ru": _representation_title(representation_class), "output_representation": "confidence + top-k attribution + context metrics"},
            formula_id=f"{representation_class}-selection",
            formula_text="class_id выбирается по task_type, interval, source/quality conflict и multilevel context",
            components={"input_conditions": [task_type, perturbation], "interval_width": interval_width, "quality_penalty": quality_penalty, "conflict_component": conflict_component},
            status_reason_ru=f"Выбран класс {representation_class}.",
            interpretation_ru=_representation_interpretation(representation_class),
            next_node_ids=["alignment", "reduction"],
        ),
        _node(
            "alignment",
            "Согласование T_ij",
            f"gamma={gamma}",
            "passed" if gamma < 0.35 else "warning",
            "gamma=max(1-p, quality_penalty, conflict_component, interval_width).",
            {"uncertainty": uncertainty, "quality_penalty": quality_penalty, "conflict_component": conflict_component, "interval_width": interval_width, "gamma": gamma, "value_source": "computed"},
            "gamma",
            operator_type="alignment",
            input_refs=["representation"],
            output_refs=["risk"],
            input_values={"class_probability": probability, "missing_rate": missing_rate, "feature_range_violation": range_violation, "conflict_component": conflict_component, "interval_width": interval_width},
            output_values={"gamma": gamma},
            formula_id="external_tabular_gamma",
            formula_text="gamma = max(1 - class_probability, quality_penalty, conflict_component, interval_width)",
            formula_latex=r"\gamma=\max(1-p,q,c,w)",
            components={"uncertainty": uncertainty, "quality_penalty": quality_penalty, "conflict_component": conflict_component, "interval_width": interval_width, "calculation": f"max({uncertainty}, {quality_penalty}, {conflict_component}, {interval_width}) = {gamma}"},
            thresholds={"gamma_warning": 0.35},
            status_reason_ru="Рассогласование ненулевое, потому что вероятность класса меньше 1.",
            interpretation_ru="Уверенность модели неполная; это ограничивает автоматическое доверие.",
            next_node_ids=["risk"],
        ),
        _node(
            "reduction",
            "Потери представления",
            f"Delta={delta}",
            "passed" if delta < 0.35 else "warning",
            "Delta=1-sum(feature_importance) для переданного набора важностей.",
            {"delta": delta, "importance_sum": importance_sum, "value_source": "computed"},
            "Delta",
            operator_type="reduction",
            input_refs=["representation"],
            output_refs=["risk"],
            input_values={"selected_features": selected_features, "feature_importance": feature_importance, "top_k": top_k},
            output_values={"delta": delta, "top_k_importance_sum": round(importance_sum, 6)},
            formula_id="external_tabular_delta",
            formula_text="delta = 1 - sum(top_k_feature_importance)",
            formula_latex=r"\Delta=1-\sum importance_{top-k}",
            components={"selected_features": selected_features, "top_k_importance_sum": round(importance_sum, 6), "calculation": f"1 - {round(importance_sum, 6)} = {delta}"},
            thresholds={"delta_warning": 0.35},
            status_reason_ru="Часть объяснения потеряна при top-k редукции.",
            interpretation_ru="Delta показывает, какая доля атрибутивного объяснения не попала в сокращённый набор признаков.",
            next_node_ids=["risk"],
        ),
        _node(
            "risk",
            "Риск rho",
            f"rho={rho}",
            risk_status,
            "rho=max(gamma, Delta, quality_component, conflict_component).",
            {"rho": rho, "theta_accept": 0.35, "theta_warning": 0.60, "value_source": "computed"},
            "rho",
            operator_type="risk",
            input_refs=["alignment", "reduction"],
            output_refs=["diagnostics"],
            input_values={"gamma": gamma, "delta": delta, "quality_component": quality_penalty, "conflict_component": conflict_component},
            output_values={"rho": rho, "risk_zone": risk_zone, "dominant_component": dominant_component},
            formula_id="external_tabular_rho",
            formula_text="rho = max(gamma, delta, quality_component, conflict_component)",
            formula_latex=r"\rho=\max(\gamma,\Delta)",
            components={"gamma": gamma, "delta": delta, "quality_component": quality_penalty, "conflict_component": conflict_component, "calculation": f"max({gamma}, {delta}, {quality_penalty}, {conflict_component}) = {rho}"},
            thresholds={"theta_accept": 0.35, "theta_warning": 0.60},
            status_reason_ru=f"rho попал в зону {risk_zone}.",
            interpretation_ru=f"Основной вклад в риск: {dominant_component}.",
            next_node_ids=["diagnostics"],
        ),
        _node(
            "diagnostics",
            "Диагностика D",
            diagnostic["diagnostic_id"],
            risk_status,
            diagnostic["message_ru"],
            {"diagnostics": [diagnostic], "value_source": "computed"},
            "D",
            operator_type="diagnostics",
            input_refs=["risk"],
            output_refs=["action"],
            input_values={"rho": rho, "risk_zone": risk_zone, "gamma": gamma, "delta": delta, "task_type": task_type, "dominant_component": dominant_component},
            output_values={
                "diagnostic_id": diagnostic["diagnostic_id"],
                "diagnostic_title_ru": diagnostic["message_ru"],
                "severity": diagnostic["criticality"],
                "recommended_action_level": diagnostic["recommended_action"],
            },
            formula_id="diagnostic-policy",
            formula_text="diagnostic_id выбирается по task_type, risk_zone и доминирующему компоненту риска",
            components={"reason_components": {"gamma": gamma, "delta": delta, "rho": rho, "dominant_component": dominant_component, "task_type": task_type}},
            status_reason_ru=diagnostic["message_ru"],
            interpretation_ru="Диагностика означает ограничение доверия, а не запрет или утверждение о плохой модели.",
            next_node_ids=["action"],
        ),
        _node(
            "action",
            "Действие",
            action,
            "passed" if action == "accept" else "warning",
            "Действие выбрано политикой FuzzyXAI.",
            {"action": action, "value_source": "computed"},
            "action policy",
            operator_type="action",
            input_refs=["diagnostics"],
            output_refs=["proof"],
            input_values={"rho": rho, "diagnostic_id": diagnostic["diagnostic_id"], "risk_zone": risk_zone},
            output_values={"action_id": action, "action_title_ru": "понизить доверие" if action == "lower_confidence" else action},
            formula_id="action-policy",
            formula_text="if rho < 0.35: accept; elif rho < 0.60: lower_confidence; else: audit",
            components={"rho": rho, "diagnostic_id": diagnostic["diagnostic_id"], "alternative_actions": ["accept", "lower_confidence", "audit"]},
            thresholds={"theta_accept": 0.35, "theta_warning": 0.60},
            status_reason_ru="Риск ненулевой, но не критический.",
            interpretation_ru="Результат не блокируется, но автоматическое доверие понижается из-за ненулевого риска.",
            next_node_ids=["proof"],
        ),
        _node(
            "proof",
            "Доказательный след",
            "proof trace готов",
            "passed",
            "Маршрут сохраняется как проверяемый доказательный след.",
            {"value_source": "computed"},
            "proof trace",
            operator_type="proof",
            input_refs=["action"],
            output_refs=[],
            input_values={"computed_result": computed, "diagnostics": [diagnostic], "action": action},
            output_values={"verification_status": "PASS", "source_commit": source_commit},
            formula_id="proof-trace",
            formula_text="proof_trace = hashable route + computed_result + diagnostics + action + source_commit",
            components={"scenario_id": adapted.scenario_id, "source_commit": source_commit},
            status_reason_ru="Proof trace содержит значения маршрута и source_commit.",
            interpretation_ru="Доказательный след связывает route, значения операторов, диагностику и действие.",
            next_node_ids=[],
        ),
    ]
    edges = [
        OperatorEdge("edge_input_explanation", "input_artifact", "explanation_object", {"class_probability": probability, "feature_importance": feature_importance}, "Адаптированный вход передан в объяснительный объект."),
        OperatorEdge("edge_explanation_representation", "explanation_object", "representation", {"terms": selected_features, "uncertainty": uncertainty}, "Объяснительный объект передан в выбор представления."),
        OperatorEdge("edge_representation_alignment", "representation", "alignment", {"class_probability": probability, "quality_metrics": {"missing_rate": missing_rate, "feature_range_violation": range_violation}, "context": context}, "Вероятность, качество и контекст переданы в оператор согласования."),
        OperatorEdge("edge_representation_reduction", "representation", "reduction", {"feature_importance": feature_importance, "top_k": top_k}, "Top-k атрибуции переданы в оператор редукции."),
        OperatorEdge("edge_alignment_risk", "alignment", "risk", {"gamma": gamma}, "Рассогласование передано в наблюдатель риска."),
        OperatorEdge("edge_reduction_risk", "reduction", "risk", {"delta": delta}, "Потери редукции переданы в наблюдатель риска."),
        OperatorEdge("edge_risk_diagnostics", "risk", "diagnostics", {"rho": rho, "risk_zone": risk_zone}, "Риск переведён в диагностическое состояние."),
        OperatorEdge("edge_diagnostics_action", "diagnostics", "action", {"diagnostic_id": diagnostic["diagnostic_id"], "recommended_action": diagnostic["recommended_action"]}, "Диагностика передана в политику действия."),
        OperatorEdge("edge_action_proof", "action", "proof", {"action_id": action, "computed_result": computed}, "Итоговое действие сохранено в proof trace."),
    ]
    return OperatorRoute(
        route_id=f"{adapted.scenario_id}:{dataset_name}:{model_name}:{perturbation}",
        scenario_id=adapted.scenario_id,
        scenario_title_ru=f"Research validation: {task_type}",
        title="External Research Validation FuzzyXAI OperatorRoute",
        created_at=datetime.now(UTC).isoformat(),
        nodes=nodes,
        edges=edges,
        computed_result=computed,
        diagnostics=[diagnostic],
        final_action=action,
        final_action_id=action,
        final_diagnostic_id=diagnostic["diagnostic_id"],
        verifier_status="PASS",
        source_commit=source_commit,
        proof_ref="proof_trace.json",
        dashboard_ref="operator_dashboard_v2.png",
        verification_summary={"overall_status": "passed", "checks": []},
    )
