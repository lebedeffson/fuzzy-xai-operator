# 3.20.X Автоматическая калибровка риск-ориентированного наблюдателя

`rho = w_p*rho_pred + w_u*u_M + w_I*(1-I_pre) + w_Delta*Delta_M + w_R*chi_R`.

Калибровка выполнена по proxy-objective на validation split. Использованы manual_config, grid_search, random_search и coordinate_search. Если несколько конфигураций имеют одинаковый `J`, применяется tie-breaker.

Автоматический поиск показал, что ручная конфигурация входит в множество оптимальных решений по proxy-objective. В итоговом ExplainPlan выбрана конфигурация с нулевой proxy-потерей и минимальной сложностью по tie-breaker.

Best config: `configs/chapter3/best_observer_config_v2.yaml`.
