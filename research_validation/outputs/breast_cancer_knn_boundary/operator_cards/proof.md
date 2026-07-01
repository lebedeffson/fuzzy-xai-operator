# Доказательный след

## Input
- `computed_result` = `{"model_name": "KNeighborsClassifier", "dataset_name": "sklearn_breast_cancer", "task_type": "tabular_classification", "perturbation": "confidence_boundary", "representation_class": "F0", "class_probability": 0.58, "uncertainty": 0.42, "quality_penalty": 0.0, "uncertainty_component": 0.42, "quality_component": 0.0, "reduction_component": 0.26, "conflict_component": 0.0, "interval_component": 0.0, "gamma": 0.42, "delta": 0.26, "rho": 0.42, "risk_dominant_component": "gamma", "diagnostic_id": "D_external_tabular_uncertainty", "action_id": "lower_confidence", "action": "lower_confidence"}`
- `diagnostics` = `[{"diagnostic_id": "D_external_tabular_uncertainty", "diagnostic_type": "external_tabular_uncertainty", "source": "external_tabular_adapter", "criticality": "medium", "message_ru": "ограниченная уверенность внешней табличной модели", "recommended_action": "lower_confidence"}]`
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
