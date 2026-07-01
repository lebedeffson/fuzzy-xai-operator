# Доказательный след

## Input
- `computed_result` = `{"model_name": "LogisticRegression", "dataset_name": "sklearn_wine", "task_type": "tabular_classification", "perturbation": "missing_features", "representation_class": "NAS", "class_probability": 0.84, "uncertainty": 0.16, "quality_penalty": 0.41, "uncertainty_component": 0.16, "quality_component": 0.41, "reduction_component": 0.24, "conflict_component": 0.0, "interval_component": 0.0, "gamma": 0.41, "delta": 0.24, "rho": 0.41, "risk_dominant_component": "gamma", "diagnostic_id": "D_external_tabular_quality_limit", "action_id": "lower_confidence", "action": "lower_confidence"}`
- `diagnostics` = `[{"diagnostic_id": "D_external_tabular_quality_limit", "diagnostic_type": "external_tabular_quality_limit", "source": "external_tabular_adapter", "criticality": "medium", "message_ru": "ограничение качества внешнего табличного входа", "recommended_action": "lower_confidence"}]`
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
