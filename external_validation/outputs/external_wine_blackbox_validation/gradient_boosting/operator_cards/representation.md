# Класс представления F

## Input
- `source_type` = `"tabular"`
- `has_single_probability` = `true`
- `has_top_k_importance` = `true`

## Formula
class_id = F0 для табличной модели с одной вероятностью и редуцированным объяснением

## Components
- `input_conditions` = `["single_probability", "top_k_feature_importance"]`

## Output
- `class_id` = `"F0"`
- `class_title_ru` = `"обычное нечёткое представление"`
- `output_representation` = `"probability + top-k feature importance"`

## Thresholds
n/a

## Status
passed: Интервальная или многоуровневая структура не требуется.

## Interpretation
F0 достаточно для внешней табличной классификации с одной вероятностью класса.

## Next
alignment, reduction
