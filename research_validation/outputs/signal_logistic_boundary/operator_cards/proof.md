# Доказательный след

## Input
- `computed_result` = `{"model_name": "LogisticRegression", "dataset_name": "synthetic_signal", "task_type": "signal_quality", "perturbation": "confidence_boundary", "representation_class": "F_ML", "class_probability": 0.57, "uncertainty": 0.43, "quality_penalty": 0.15, "uncertainty_component": 0.43, "quality_component": 0.15, "reduction_component": 0.25, "conflict_component": 0.0, "interval_component": 0.0, "gamma": 0.43, "delta": 0.25, "rho": 0.43, "risk_dominant_component": "gamma", "diagnostic_id": "D_external_tabular_uncertainty", "action_id": "lower_confidence", "action": "lower_confidence"}`
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
