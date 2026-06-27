# Layered Demo Guide (RU)

## Запуск

```bash
make layered-demo PORT=8096
```

## Что показывает каждый слой

1. Вход и модель: объект, признаки, `prediction`, `predicted_risk`.
2. Оператор `Omega`: `E_model=<L,mu,R,alpha,u,tau>`, граф функций принадлежности.
3. Категория/HoTT: переходы `E_model -> E_risk -> E_action`, `morphism` или `rupture`.
4. Иерархия неопределённости: профиль `u_*`, выбор класса и `Delta`.
5. Топос-контекст: `RiskContext`, `AutoAccept`, интерпретация `chi_Auto`.
6. Риск-наблюдатель: формула `rho`, вклады компонентов, пороговая политика с `chi_R` и `chi_R^crit`.

## Как читать результат

- `chi_R^crit=1` => `block` независимо от `rho`.
- `rho>=0.80` и `chi_R^crit=0` => `defer_to_human`.
- `MISSING` у registry-режима = нет локального файла, не ошибка метода.

## Экспорт

- Кнопка `Export trace JSON` в GUI.
- Файл: `reports/layered_demo/last_case_trace.json`.

## Что показывать на защите

1. Переключение между `breast_cancer` и `synthetic_ruptures`.
2. Демонстрация кейса с `rupture`.
3. Вклад компонентов в `rho` и итоговое действие.
4. Экспорт трассы в JSON.
