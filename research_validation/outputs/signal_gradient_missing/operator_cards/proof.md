# Доказательный след

## Input
- `computed_result` = `{"model_name": "GradientBoostingClassifier", "dataset_name": "synthetic_signal", "task_type": "signal_quality", "perturbation": "missing_features", "representation_class": "F_ML", "class_probability": 0.8, "uncertainty": 0.2, "quality_penalty": 0.73, "uncertainty_component": 0.2, "quality_component": 0.73, "reduction_component": 0.32, "conflict_component": 0.0, "interval_component": 0.0, "gamma": 0.73, "delta": 0.32, "rho": 0.73, "risk_dominant_component": "gamma", "diagnostic_id": "D_signal_missing_fragments", "action_id": "defer_to_human", "action": "defer_to_human"}`
- `diagnostics` = `[{"diagnostic_id": "D_signal_missing_fragments", "diagnostic_type": "signal_missing_fragments", "source": "external_tabular_adapter", "criticality": "high", "message_ru": "ограничение качества сигнала", "recommended_action": "defer_to_human"}]`
- `action` = `"defer_to_human"`

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
