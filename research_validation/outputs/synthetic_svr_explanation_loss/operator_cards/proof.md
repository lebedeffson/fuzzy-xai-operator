# Доказательный след

## Input
- `computed_result` = `{"model_name": "SVR", "dataset_name": "synthetic_regression", "task_type": "tabular_regression", "perturbation": "top_k_reduction", "representation_class": "F_int", "class_probability": 0.71, "uncertainty": 0.29, "quality_penalty": 0.0, "uncertainty_component": 0.29, "quality_component": 0.0, "reduction_component": 0.55, "conflict_component": 0.0, "interval_component": 0.3, "gamma": 0.3, "delta": 0.55, "rho": 0.55, "risk_dominant_component": "delta", "diagnostic_id": "D_external_regression_explanation_loss", "action_id": "lower_confidence", "action": "lower_confidence"}`
- `diagnostics` = `[{"diagnostic_id": "D_external_regression_explanation_loss", "diagnostic_type": "external_regression_explanation_loss", "source": "external_tabular_adapter", "criticality": "medium", "message_ru": "ограниченная уверенность внешней регрессионной модели", "recommended_action": "lower_confidence"}]`
- `action` = `"lower_confidence"`

## Formula
proof_trace = hashable route + computed_result + diagnostics + action + source_commit

## Components
- `scenario_id` = `"external_wine_classification"`
- `source_commit` = `"11955855fb111912239f83500af5349fba895ee5"`

## Output
- `verification_status` = `"PASS"`
- `source_commit` = `"11955855fb111912239f83500af5349fba895ee5"`

## Thresholds
n/a

## Status
passed: Proof trace содержит значения маршрута и source_commit.

## Interpretation
Доказательный след связывает route, значения операторов, диагностику и действие.

## Next
final
