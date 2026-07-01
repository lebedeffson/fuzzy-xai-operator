# Доказательный след

## Input
- `computed_result` = `{"class_probability": 0.689724, "uncertainty": 0.310276, "quality_penalty": 0.0, "gamma": 0.310276, "delta": 0.373128, "rho": 0.373128, "diagnostic_id": "D_external_tabular_uncertainty", "action_id": "lower_confidence", "action": "lower_confidence"}`
- `diagnostics` = `[{"diagnostic_id": "D_external_tabular_uncertainty", "diagnostic_type": "external_tabular_uncertainty", "source": "external_tabular_adapter", "criticality": "medium", "message_ru": "ограниченная уверенность внешней табличной модели", "recommended_action": "lower_confidence"}]`
- `action` = `"lower_confidence"`

## Formula
proof_trace = hashable route + computed_result + diagnostics + action + source_commit

## Components
- `scenario_id` = `"external_wine_classification"`
- `source_commit` = `"be2a30096b275d3365a92cb69b165e57e99e90de"`

## Output
- `verification_status` = `"PASS"`
- `source_commit` = `"be2a30096b275d3365a92cb69b165e57e99e90de"`

## Thresholds
n/a

## Status
passed: Proof trace содержит значения маршрута и source_commit.

## Interpretation
Доказательный след связывает route, значения операторов, диагностику и действие.

## Next
final
