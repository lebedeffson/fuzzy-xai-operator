# Доказательный след

## Input
- `computed_result` = `{"model_name": "LogisticRegression", "dataset_name": "sklearn_wine", "task_type": "tabular_classification", "perturbation": "external_payload", "representation_class": "F0", "class_probability": 0.689724, "uncertainty": 0.310276, "quality_penalty": 0.0, "uncertainty_component": 0.310276, "quality_component": 0.0, "presentation_omission_component": 0.373128, "conflict_component": 0.0, "interval_component": 0.0, "legacy_route_gap": 0.310276, "presentation_omission_loss": 0.373128, "legacy_route_score": 0.373128, "scientific_contract": "legacy_external_route_metrics_not_P19_Gamma_Delta_rho", "risk_dominant_component": "presentation_omission", "diagnostic_id": "D_external_tabular_uncertainty", "action_id": "lower_confidence", "action": "lower_confidence"}`
- `diagnostics` = `[{"diagnostic_id": "D_external_tabular_uncertainty", "diagnostic_type": "external_tabular_uncertainty", "source": "external_tabular_adapter", "criticality": "medium", "message_ru": "ограниченная уверенность внешней табличной модели", "recommended_action": "lower_confidence"}]`
- `action` = `"lower_confidence"`

## Formula
proof_trace = hashable route + computed_result + diagnostics + action + source_commit

## Components
- `scenario_id` = `"external_wine_classification"`
- `source_commit` = `"97074b57a5522cb1cd7150f288a8b27bd915b357"`

## Output
- `verification_status` = `"PASS"`
- `source_commit` = `"97074b57a5522cb1cd7150f288a8b27bd915b357"`

## Thresholds
n/a

## Status
passed: Proof trace содержит значения маршрута и source_commit.

## Interpretation
Доказательный след связывает route, значения операторов, диагностику и действие.

## Next
final
