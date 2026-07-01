from __future__ import annotations


OPERATORS: dict[str, dict[str, object]] = {
    "input_artifact": {
        "operator_id": "input_artifact",
        "title_ru": "Входной артефакт",
        "input_contract": ["payload"],
        "output_contract": ["AdaptedInput"],
        "formula_id": None,
        "formula_text": "Нормализация внешнего входа",
        "required_components": ["model_output", "quality_metrics"],
        "produced_values": ["adapted_input"],
    },
    "explanation_object": {
        "operator_id": "explanation_object",
        "title_ru": "Объяснительный объект",
        "input_contract": ["AdaptedInput"],
        "output_contract": ["E_k"],
        "formula_id": "E_k",
        "formula_text": "E_k = {L_k, μ_k, R_k, α_k, u_k, τ_k}",
        "required_components": ["feature_values", "feature_importance"],
        "produced_values": ["terms", "memberships", "uncertainty"],
    },
    "representation": {
        "operator_id": "representation",
        "title_ru": "Класс представления",
        "input_contract": ["E_k", "context_values"],
        "output_contract": ["representation_class"],
        "formula_id": "F",
        "formula_text": "F = policy(E_k, context)",
        "required_components": ["task_type", "quality_component", "interval_component"],
        "produced_values": ["representation_class"],
    },
    "alignment": {
        "operator_id": "alignment",
        "title_ru": "Согласование",
        "input_contract": ["class_probability", "quality_metrics"],
        "output_contract": ["gamma"],
        "formula_id": "gamma_external_tabular",
        "formula_text": "gamma = max(1 - p, quality_penalty, conflict, interval)",
        "required_components": ["uncertainty_component", "quality_component"],
        "produced_values": ["gamma"],
    },
    "reduction": {
        "operator_id": "reduction",
        "title_ru": "Редукция",
        "input_contract": ["feature_importance"],
        "output_contract": ["delta"],
        "formula_id": "delta_top_k",
        "formula_text": "delta = 1 - sum(top_k_feature_importance)",
        "required_components": ["feature_importance"],
        "produced_values": ["delta"],
    },
    "risk": {
        "operator_id": "risk",
        "title_ru": "Риск",
        "input_contract": ["gamma", "delta"],
        "output_contract": ["rho", "risk_zone"],
        "formula_id": "rho_max",
        "formula_text": "rho = max(gamma, delta, quality, conflict, interval)",
        "required_components": ["gamma", "delta"],
        "produced_values": ["rho", "risk_dominant_component"],
    },
    "diagnostics": {
        "operator_id": "diagnostics",
        "title_ru": "Диагностика",
        "input_contract": ["rho", "dominant_component"],
        "output_contract": ["diagnostic_id"],
        "formula_id": "D",
        "formula_text": "D = diagnostic_policy(rho, components)",
        "required_components": ["rho", "task_type"],
        "produced_values": ["diagnostic_id"],
    },
    "action": {
        "operator_id": "action",
        "title_ru": "Действие",
        "input_contract": ["rho", "diagnostic_id"],
        "output_contract": ["action_id"],
        "formula_id": "A",
        "formula_text": "A = action_policy(rho, diagnostic)",
        "required_components": ["rho", "diagnostic_id"],
        "produced_values": ["action_id"],
    },
    "proof": {
        "operator_id": "proof",
        "title_ru": "Доказательный след",
        "input_contract": ["route"],
        "output_contract": ["proof_trace"],
        "formula_id": None,
        "formula_text": "Фиксация route, computed_result и verifier checks",
        "required_components": ["source_commit", "computed_result"],
        "produced_values": ["proof_trace", "verifier_report"],
    },
}


def list_operators() -> list[dict[str, object]]:
    return [OPERATORS[key] for key in sorted(OPERATORS)]


def get_operator(operator_id: str) -> dict[str, object]:
    return OPERATORS[operator_id]
