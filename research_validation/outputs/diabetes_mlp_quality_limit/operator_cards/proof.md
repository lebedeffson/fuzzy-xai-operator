# Доказательный след

## Input
- `computed_result` = `{"model_name": "MLPRegressor", "dataset_name": "sklearn_diabetes", "task_type": "tabular_regression", "perturbation": "missing_features", "representation_class": "NAS", "class_probability": 0.74, "uncertainty": 0.26, "quality_penalty": 0.52, "uncertainty_component": 0.26, "quality_component": 0.52, "reduction_component": 0.28, "conflict_component": 0.0, "interval_component": 0.36, "gamma": 0.52, "delta": 0.28, "rho": 0.52, "risk_dominant_component": "gamma", "diagnostic_id": "D_external_regression_uncertainty", "action_id": "lower_confidence", "action": "lower_confidence"}`
- `diagnostics` = `[{"diagnostic_id": "D_external_regression_uncertainty", "diagnostic_type": "external_regression_uncertainty", "source": "external_tabular_adapter", "criticality": "medium", "message_ru": "ограниченная уверенность внешней регрессионной модели", "recommended_action": "lower_confidence"}]`
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
