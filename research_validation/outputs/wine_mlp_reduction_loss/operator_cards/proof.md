# Доказательный след

## Input
- `computed_result` = `{"model_name": "MLPClassifier", "dataset_name": "sklearn_wine", "task_type": "tabular_classification", "perturbation": "top_k_reduction", "representation_class": "F_ML", "class_probability": 0.76, "uncertainty": 0.24, "quality_penalty": 0.0, "uncertainty_component": 0.24, "quality_component": 0.0, "reduction_component": 0.52, "conflict_component": 0.0, "interval_component": 0.0, "gamma": 0.24, "delta": 0.52, "rho": 0.52, "risk_dominant_component": "delta", "diagnostic_id": "D_external_tabular_reduction_loss", "action_id": "lower_confidence", "action": "lower_confidence"}`
- `diagnostics` = `[{"diagnostic_id": "D_external_tabular_reduction_loss", "diagnostic_type": "external_tabular_reduction_loss", "source": "external_tabular_adapter", "criticality": "medium", "message_ru": "потери редуцированного табличного объяснения", "recommended_action": "lower_confidence"}]`
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
