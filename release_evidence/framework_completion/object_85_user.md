## Итог
- Модель сформировала прогноз [0]. Максимальный модельный балл равен 0.779. Рекомендуемое действие: review.

## Главные причины
- Правило linear_0_0 (суррогатное): marker_a has a higher linear contribution to class 0; значимость 2.032.
- Правило linear_0_1 (суррогатное): marker_b has a higher linear contribution to class 0; значимость 2.022.
- Правило R_rare_subtype_surrogate (суррогатное): rare positive subtype identified by the monitored marker region; значимость 0.498.

## Что модель увидела
- Качество входных данных объекта 85: 1.000.
- Объект 85 впервые устойчиво распознан на эпохе 0.
- Class 0: marker_a has a higher linear contribution to class 0; marker_b has a higher linear contribution to class 0; coverage unavailable.

## Что потеряно или усреднено
- Для объекта 85 обнаружены события забывания на эпохах [16].
- При росте общей метрики на +0.004 метрика подгруппы rare_positive изменилась на -1.000.

## Похожие случаи
- Объект 220: сходство 1.000 по методу robust_standardized_euclidean; сравнивались normalized tabular feature vector.
- Объект 225: сходство 0.996 по методу robust_standardized_euclidean; сравнивались normalized tabular feature vector.

## Что изменило бы решение
- Изменение признаков {'marker_b': {'from': 0.23943944027389774, 'to': 0.9385666402614603}} переводит прогноз из 0 в 1; наблюдаемый эффект -0.240557.
- Изменение признаков {'marker_b': {'from': 0.23943944027389774, 'to': 1.2778269282218877}} переводит прогноз из 0 в 1; наблюдаемый эффект -0.080536.

## Доверие
- Все заявленные факты связаны с узлами ExplanationGraph.

## Ограничения
- embedding spread unavailable
- primary-rule coverage is unavailable
- feature distance does not establish causal or clinical similarity
- quantile search tests association, not causal feasibility
- overall test accuracy changed by -0.004386
- controlled synthetic protocol; effect must be re-measured for another dataset
