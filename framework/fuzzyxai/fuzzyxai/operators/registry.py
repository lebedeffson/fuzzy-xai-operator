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
        "input_contract": ["SituationProfile", "Candidate[]"],
        "output_contract": ["Candidate", "D_choice"],
        "formula_id": "P_sit/D_choice",
        "formula_text": "minimal sufficient class on the admissible Pareto front",
        "required_components": ["profile", "candidate_coverage", "complexity", "expected_reduction_loss"],
        "produced_values": ["selected_candidate", "choice_diagnostic"],
    },
    "alignment": {
        "operator_id": "alignment",
        "title_ru": "Согласование",
        "input_contract": ["AlignmentInput"],
        "output_contract": ["AlignmentResult"],
        "formula_id": "d_E/gamma/T_ij",
        "formula_text": "gamma = weighted semantic disagreement of supplied components",
        "required_components": ["components", "weights", "gamma_max", "delta_t", "delta_max"],
        "produced_values": ["gamma", "certified"],
    },
    "reduction": {
        "operator_id": "reduction",
        "title_ru": "Редукция",
        "input_contract": ["ReductionInput"],
        "output_contract": ["ReductionOperatorResult"],
        "formula_id": "Delta",
        "formula_text": "Delta = weighted loss of supplied reduction components",
        "required_components": ["components", "weights", "delta_max"],
        "produced_values": ["delta", "r_delta", "allowed"],
    },
    "risk": {
        "operator_id": "risk",
        "title_ru": "Риск",
        "input_contract": ["RiskInput"],
        "output_contract": ["RiskResult"],
        "formula_id": "rho/chi_R_crit",
        "formula_text": "rho = weighted risk components; action = policy(rho, chi_R_crit)",
        "required_components": ["components", "weights", "thresholds", "chi_r_crit"],
        "produced_values": ["rho", "chi_r_crit", "action"],
    },
    "diagnostics": {
        "operator_id": "diagnostics",
        "title_ru": "Диагностика",
        "input_contract": ["condition_map", "DiagnosticRule[]"],
        "output_contract": ["DiagnosticState[]"],
        "formula_id": "D",
        "formula_text": "D = declared diagnostic rules satisfied by observed conditions",
        "required_components": ["conditions", "rules"],
        "produced_values": ["diagnostic_states"],
    },
    "action": {
        "operator_id": "action",
        "title_ru": "Действие",
        "input_contract": ["RiskResult", "DiagnosticState[]"],
        "output_contract": ["ActionResult"],
        "formula_id": "A",
        "formula_text": "A = action_policy(RiskResult, diagnostics)",
        "required_components": ["risk_result", "diagnostics"],
        "produced_values": ["action", "status", "reason"],
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
