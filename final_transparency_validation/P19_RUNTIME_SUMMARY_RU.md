# P19 — итог runtime closure

Полный системный маршрут вычисляется одним публичным вызовом:

`FuzzyXAI.wrap(..., observation_context=...).explain_one(...)`.

После вызова typed `ModelExplanationResult.system` содержит `E_model`,
исполненный `T_ij`, согласованный объект, `E_target`, Gamma, три канала
неопределённости, `u_M`, `F_int`, редукцию и Delta, `E_pre`, `I_pre`, полную
пятикомпонентную rho, critical override и action. System generators после
`explain_one()` только экспортируют этот результат. Image generator отдельно
получает caller-defined connected regions из публичного attribution tensor;
он не вычисляет системные поля или IG completeness повторно.

| case | Gamma | U_model | U_rules | U_trace | u_M | Delta | I_pre | rho | action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| system accept | 0 | 0 | 0 | 0 | 0 | 0 | 0.8962481423 | 0.0207503715 | accept |
| system conflict | 0.05 | 0 | 0 | 1 | 0.2 | 0 | 0.8611057500 | 0.1777788500 | block |
| system reduction | 0.3334630325 | 0.4987734957 | 0.1177083333 | 0 | 0.2846992478 | 0.4817304978 | 0.8077111390 | 0.3424859088 | accept |

Conflict — контролируемая fault-injection: неверифицируемая target trace
намеренно передана через `ObservationContext`. Численная rho остаётся полной;
block задаётся отдельным critical override.

Same-run training: `SGDClassifier.partial_fit`, run
`sgd-bcw-p19-run-19`, fingerprint
`510386e38ad7319ce15399026dec514fbdcef74a8b3a954e4cec79b15c3a7e68`,
first learned epoch 1, correctness-transition forgetting at epoch 27,
stability 0.965517, measured loss.

Reduction-case использует реальный held-out object 74: 107/93 голосов деревьев,
hard-vote proportion риска 0.535 и model probability 0.5353125. В rho_p входит
model probability; голоса отдельно дают `U_model=0.4987734957`. Интервал
`[0.0365390043, 1]`, midpoint 0.5182695022 и положительная
`Delta=0.4817304978`. Значения не задаются генератором.

Image IG: target 0, zero baseline, logit delta 9.9367974997. При 512
трапецеидальных интервалах attribution sum=9.9352016449, absolute residual
0.0015958548 (relative 0.0001606005). Сохранён convergence audit для
16/32/64/128/256/512 шагов.

Golden tabular не имеет полной пятикомпонентной системы: `rho=None`,
`risk_status=incomplete`; старое локальное число сохранено только как
`partial_risk_score=0.104253`.

Глобальная consistency closure: ExplainPlan управляет применимостью T_ij,
методом U_model и реальной редукцией; несовпадающий context transform закрывает
route fail-closed. RF votes не создают `local_contributions`. Внешний legacy
route экспортирует `presentation_omission_loss` и `legacy_route_score`, но не
канонические Gamma/Delta/rho. Public `observe_risk` принимает только пять
компонент P19 без неявной перенормировки.

Финальные ворота: full regression — 1261 passed, 11 skipped, 643 warnings
за 288.12 s; scoped mypy и ruff — без ошибок; wheel-only smoke вне source
checkout — PASS. Manifest содержит 41 оператор и загружается из package data;
проверены local, RF-system и non-RF-system public routes. Базовая установка не
ставит NiceGUI. Namespace `fuzzyxai.experiments` сохранён, поскольку семь
защищённых manifest-операторов реально импортируют оттуда callables; шесть
отсоединённых research namespaces исключены.

Финальный review bundle содержит 1 010 payload-файлов и 1 012 файлов вместе
с программно созданными `BUNDLE_CONTENTS.txt` и `SHA256SUMS.txt`.
