# Доказательный след

## Input
- `computed_result` = `{"model_name": "MLPClassifier", "dataset_name": "sklearn_digits", "task_type": "image_like_classification", "perturbation": "noise", "representation_class": "F_ML", "class_probability": 0.66, "uncertainty": 0.34, "quality_penalty": 0.42, "uncertainty_component": 0.34, "quality_component": 0.42, "reduction_component": 0.41, "conflict_component": 0.0, "interval_component": 0.0, "gamma": 0.42, "delta": 0.41, "rho": 0.42, "risk_dominant_component": "gamma", "diagnostic_id": "D_external_image_uncertainty", "action_id": "lower_confidence", "action": "lower_confidence"}`
- `diagnostics` = `[{"diagnostic_id": "D_external_image_uncertainty", "diagnostic_type": "external_image_uncertainty", "source": "external_tabular_adapter", "criticality": "medium", "message_ru": "ограничение image-like объяснения", "recommended_action": "lower_confidence"}]`
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
