# Доказательный след

## Input
- `computed_result` = `{"model_name": "threshold_model", "dataset_name": "synthetic_signal", "task_type": "signal_quality", "perturbation": "clean", "representation_class": "F_ML", "class_probability": 0.78, "uncertainty": 0.22, "quality_penalty": 0.08, "uncertainty_component": 0.22, "quality_component": 0.08, "reduction_component": 0.22, "conflict_component": 0.0, "interval_component": 0.0, "gamma": 0.22, "delta": 0.22, "rho": 0.22, "risk_dominant_component": "gamma", "diagnostic_id": "D_external_tabular_ok", "action_id": "accept", "action": "accept"}`
- `diagnostics` = `[{"diagnostic_id": "D_external_tabular_ok", "diagnostic_type": "external_tabular_ok", "source": "external_tabular_adapter", "criticality": "low", "message_ru": "внешний результат допустим", "recommended_action": "accept"}]`
- `action` = `"accept"`

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
