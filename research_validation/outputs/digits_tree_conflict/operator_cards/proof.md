# Доказательный след

## Input
- `computed_result` = `{"model_name": "DecisionTreeClassifier", "dataset_name": "sklearn_digits", "task_type": "tabular_classification", "perturbation": "explanation_conflict", "representation_class": "NAS", "class_probability": 0.83, "uncertainty": 0.17, "quality_penalty": 0.0, "uncertainty_component": 0.17, "quality_component": 0.0, "reduction_component": 0.29, "conflict_component": 0.67, "interval_component": 0.0, "gamma": 0.67, "delta": 0.29, "rho": 0.67, "risk_dominant_component": "gamma", "diagnostic_id": "D_rule_attribution_conflict", "action_id": "audit", "action": "audit"}`
- `diagnostics` = `[{"diagnostic_id": "D_rule_attribution_conflict", "diagnostic_type": "rule_attribution_conflict", "source": "external_tabular_adapter", "criticality": "high", "message_ru": "конфликт объяснительного источника и модельного сигнала", "recommended_action": "audit"}]`
- `action` = `"audit"`

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
