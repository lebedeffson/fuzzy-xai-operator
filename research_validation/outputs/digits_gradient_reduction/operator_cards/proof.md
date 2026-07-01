# Доказательный след

## Input
- `computed_result` = `{"model_name": "GradientBoostingClassifier", "dataset_name": "sklearn_digits", "task_type": "image_like_classification", "perturbation": "top_k_reduction", "representation_class": "F_ML", "class_probability": 0.7, "uncertainty": 0.3, "quality_penalty": 0.12, "uncertainty_component": 0.3, "quality_component": 0.12, "reduction_component": 0.5, "conflict_component": 0.0, "interval_component": 0.0, "gamma": 0.3, "delta": 0.5, "rho": 0.5, "risk_dominant_component": "delta", "diagnostic_id": "D_image_explanation_reduction", "action_id": "lower_confidence", "action": "lower_confidence"}`
- `diagnostics` = `[{"diagnostic_id": "D_image_explanation_reduction", "diagnostic_type": "image_explanation_reduction", "source": "external_tabular_adapter", "criticality": "medium", "message_ru": "ограничение image-like объяснения", "recommended_action": "lower_confidence"}]`
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
