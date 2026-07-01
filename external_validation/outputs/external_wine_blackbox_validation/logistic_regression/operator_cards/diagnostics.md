# Диагностика D

## Input
- `rho` = `0.373128`
- `risk_zone` = `"lower_confidence"`
- `gamma` = `0.310276`
- `delta` = `0.373128`

## Formula
diagnostic_id выбирается по risk_zone и ненулевым компонентам gamma/delta

## Components
- `reason_components` = `{"gamma": 0.310276, "delta": 0.373128, "rho": 0.373128}`

## Output
- `diagnostic_id` = `"D_external_tabular_uncertainty"`
- `diagnostic_title_ru` = `"ограниченная уверенность внешней табличной модели"`
- `severity` = `"medium"`
- `recommended_action_level` = `"lower_confidence"`

## Thresholds
n/a

## Status
warning: ограниченная уверенность внешней табличной модели

## Interpretation
Диагностика означает ограничение доверия, а не запрет или утверждение о плохой модели.

## Next
action
