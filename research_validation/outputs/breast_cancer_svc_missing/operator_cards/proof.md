# Доказательный след

## Input
- `computed_result` = `{"model_name": "SVC(probability=True)", "dataset_name": "sklearn_breast_cancer", "task_type": "tabular_classification", "perturbation": "missing_features", "representation_class": "NAS", "class_probability": 0.72, "uncertainty": 0.28, "quality_penalty": 0.62, "uncertainty_component": 0.28, "quality_component": 0.62, "reduction_component": 0.3, "conflict_component": 0.0, "interval_component": 0.0, "gamma": 0.62, "delta": 0.3, "rho": 0.62, "risk_dominant_component": "gamma", "diagnostic_id": "D_external_tabular_quality_limit", "action_id": "audit", "action": "audit"}`
- `diagnostics` = `[{"diagnostic_id": "D_external_tabular_quality_limit", "diagnostic_type": "external_tabular_quality_limit", "source": "external_tabular_adapter", "criticality": "high", "message_ru": "ограничение качества внешнего табличного входа", "recommended_action": "audit"}]`
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
