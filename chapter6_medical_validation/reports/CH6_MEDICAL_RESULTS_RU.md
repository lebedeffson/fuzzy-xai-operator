# Глава 6 — практические результаты medical validation

## Дизайн

Выполнены две реальные open-data public-runtime validations: PTB-XL (12-lead ECG) и Allen CCF 2017 (`brain_v2_confirmatory`, section-block-wise Nissl patches). `brain_v1_pilot` сохранён как исторический малый pilot и не заменяет confirmatory route. Офтальмологическая линия IDRiD подготовлена как primary continuation, но не выполнена: официальный IEEE DataPort требует аутентифицированного принятия условий, поэтому метрики, XAI и system results для глаз не фабрикуются. Следовательно, текущий результат — две выполненные validation и одна подготовленная application line, а не three-domain empirical claim.

## Что измерялось

Для каждого выполненного domain public `FuzzyXAI.wrap(...).explain_one(...)` возвращал prediction, model-native evidence, typed evidence, `system_Gamma`, uncertainty profile, action, audit и directed provenance. `system_Gamma` обозначает canonical alignment зарегистрированных system interfaces; он не является agreement между двумя native XAI maps. ECG IG/temporal occlusion и brain Grad-CAM/IG сравнения остаются отдельными diagnostic quantities. Когда ExplainPlan указывает `reduction=not_applied`, chapter tables показывают Δ как «не применялось», а не как измеренную нулевую loss.

## Representative outcomes и controls

ECG и brain reports содержат real registered test splits, model metrics, selected cases, native XAI and public system artifacts. Controlled checkpoint/provenance/preprocessing faults были обнаружены как critical integrity conditions и fail-closed блокировались. Это не означает, что framework выявляет все ошибки модели: в ECG уверенные ложноположительный и ложноотрицательный predictions могут сохранять internally consistent route и получать technical `accept`, если ExplainPlan не содержит внешнего verification evidence.

## Ограничения

Техническое action policy не является clinical decision support; native attribution не доказывает биологическую причинность; Allen v2 ограничен single-atlas section-block generalization. Все сильные claims ограничены зарегистрированными данными, ExplainPlan и provenance конкретного route.
