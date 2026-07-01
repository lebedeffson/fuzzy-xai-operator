# Доказательный след

## Input
- `computed_result` = `{"model_name": "GradientBoostingClassifier", "dataset_name": "sklearn_wine", "task_type": "tabular_classification", "perturbation": "external_payload", "representation_class": "F0", "class_probability": 0.679119, "uncertainty": 0.320881, "quality_penalty": 0.0, "uncertainty_component": 0.320881, "quality_component": 0.0, "reduction_component": 0.429844, "conflict_component": 0.0, "interval_component": 0.0, "gamma": 0.320881, "delta": 0.429844, "rho": 0.429844, "risk_dominant_component": "delta", "diagnostic_id": "D_external_tabular_uncertainty", "action_id": "lower_confidence", "action": "lower_confidence"}`
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
