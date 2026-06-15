# Вставка: автоматическая калибровка наблюдателя

Калибровка выполнена по proxy-objective, а не по клинической экспертной разметке.

`rho = w_p*rho_pred + w_u*u_M + w_I*(1-I_pre) + w_Delta*Delta_M + w_R*chi_R`.

Целевая функция:

`J = 5*missed_critical_ruptures + 3*false_auto_accept + 2*unsafe_accept_with_conflict + false_block + 0.5*excessive_defer`.

Лучшие параметры сохранены в `configs/chapter3/best_observer_config.yaml`:

```yaml
weights:
  w_p: 0.35
  w_u: 0.15
  w_I: 0.2
  w_Delta: 0.1
  w_R: 0.2
thresholds:
  - 0.25
  - 0.45
  - 0.65
  - 0.82
gamma_max: 0.5
I_min: 0.4
Delta_max: 0.35

```

Выбранные параметры должны фиксироваться в `ExplainPlan` и trace маршрута.
