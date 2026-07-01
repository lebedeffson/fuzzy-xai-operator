# Доказательный след

## Input
- `computed_result` = `{"model_name": "LogisticRegression", "dataset_name": "synthetic_signal", "task_type": "signal_quality", "perturbation": "noise", "representation_class": "F_ML", "class_probability": 0.75, "uncertainty": 0.25, "quality_penalty": 0.61, "uncertainty_component": 0.25, "quality_component": 0.61, "reduction_component": 0.28, "conflict_component": 0.0, "interval_component": 0.0, "gamma": 0.61, "delta": 0.28, "rho": 0.61, "risk_dominant_component": "gamma", "diagnostic_id": "D_signal_noise_limit", "action_id": "defer_to_human", "action": "defer_to_human"}`
- `diagnostics` = `[{"diagnostic_id": "D_signal_noise_limit", "diagnostic_type": "signal_noise_limit", "source": "external_tabular_adapter", "criticality": "high", "message_ru": "ограничение качества сигнала", "recommended_action": "defer_to_human"}]`
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
