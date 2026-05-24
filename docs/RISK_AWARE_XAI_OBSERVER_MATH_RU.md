# Risk-Aware XAI Observer: математическая модель

Этот слой является надстройкой над главами 2 и 3. Он не меняет параметры модели, а наблюдает за прогнозом, строит расширенное объяснение, оценивает риск автоматического применения и выбирает безопасное действие.

## 1. Прогнозный интерфейс модели

Пусть дана модель:

\[
M:X\to Y.
\]

Для классификации:

\[
M(x)=\mathbf p(x)=(p_1(x),\dots,p_K(x)),\qquad \sum_{k=1}^{K}p_k(x)=1.
\]

Для бинарного риска:

\[
r_M(x)=p_1(x).
\]

## 2. Расширенный объяснительный объект

Системный оператор главы 2 строит:

\[
E_M=\langle L_M,\mu_M,R_M,\alpha_M,u_M,\tau_M\rangle.
\]

С учётом иерархии главы 3 обычные функции принадлежности заменяются выбранным представлением неопределённости:

\[
E_M^{ext}=\langle L_M,A_M^{\mathcal F},R_M,\alpha_M,u_M,\tau_M,\Delta_M\rangle.
\]

Здесь `A_M^F` - минимально достаточное представление из иерархии `F0 / IntervalFS / HesitantFS / NeutrosophicFS / MultiLevelFS`, а `Delta_M` - потеря редукции.

## 3. Модельная неопределённость

Энтропийная неопределённость:

\[
u_{ent}(x)=-\frac{1}{\log K}\sum_{k=1}^{K}p_k(x)\log(p_k(x)+\varepsilon).
\]

Маржинальная неопределённость:

\[
u_{mar}(x)=1-(p_{(1)}(x)-p_{(2)}(x)).
\]

Агрегация:

\[
u_M^0(x)=\omega_{ent}u_{ent}(x)+\omega_{mar}u_{mar}(x),\qquad \omega_{ent}+\omega_{mar}=1.
\]

В коде это реализовано в `fuzzyxai/risk/uncertainty.py`.

## 4. Риск автоматического применения прогноза

Оценивается не только риск класса, а риск автоматического использования прогноза:

\[
\rho(x)=w_p\rho_p(x)+w_u u_M(x)+w_I(1-I(E_G))+w_\Delta\Delta_M+w_D\mathbf 1_{D\neq\varnothing}.
\]

Смысл компонент:

- `rho_p(x)` - собственный риск прогноза;
- `u_M(x)` - неопределённость модели;
- `1 - I(E_G)` - потеря системной интерпретируемости;
- `Delta_M` - потеря редукции из главы 3;
- `1[D != empty]` - штраф за диагностический разрыв.

Веса неотрицательны и нормируются на симплекс. В коде формула вынесена в `fuzzyxai/risk/risk_function.py`.

## 5. Действия наблюдателя

\[
\mathcal A=\{accept, lower\_confidence, request\_more\_data, defer\_to\_human, block\}.
\]

Пороговая политика:

\[
a^*(x)=
\begin{cases}
block, & D\neq\varnothing,\\
defer\_to\_human, & \rho(x)\ge \theta_{high},\\
request\_more\_data, & \theta_{mid}\le \rho(x)<\theta_{high},\\
lower\_confidence, & u_M(x)\ge \theta_u \land r_M(x)\ge\theta_r,\\
accept, & \rho(x)<\theta_{mid}.
\end{cases}
\]

В коде это `RiskPolicy` из `fuzzyxai/risk/decision_policy.py`.

## 6. Оптимальная политика через стоимость

Для более строгой постановки задаётся матрица стоимости `C(a,y)`:

\[
\bar C(a\mid x,E_M^{ext})=\sum_y \hat P(y\mid x)C(a,y).
\]

Оптимальное действие:

\[
a^*(x)=\arg\min_{a\in\mathcal A}\bar C(a\mid x,E_M^{ext}).
\]

В коде базовые функции: `expected_action_costs` и `choose_min_expected_cost`.

## 7. Композиция объектов

Строятся три объяснительных объекта:

\[
E_M^{ext},\qquad E_R,\qquad E_D.
\]

Итоговая цепочка:

\[
E_G=E_D\odot E_R\odot E_M^{ext}.
\]

Рассогласования:

\[
\gamma_{MR}=d_E^{ext}(E_M^{ext},E_R),\qquad \gamma_{RD}=d_E^{ext}(E_R,E_D).
\]

Индекс:

\[
I(E_G)=\exp(-\mathcal L(E_G)).
\]

Если согласование невозможно, возвращается диагностическое состояние `D_ij`, а автоматическое решение блокируется.

## 8. Утверждение о снижении оценочного риска

Если политика выбирает

\[
a^*(x)=\arg\min_{a\in\mathcal A}\bar C(a\mid x),
\]

а `accept` входит в множество допустимых действий, то:

\[
\bar C(a^*(x)\mid x)\le \bar C(accept\mid x).
\]

Доказательство: минимум по множеству действий не больше стоимости любого конкретного действия из этого множества. Утверждение относится к оценочной функции стоимости, а не обещает рост реальной точности без калибровки.

## 9. Алгоритм

1. Получить `predict_proba(x)`.
2. Вычислить `u_ent`, `u_mar`, `u_M`.
3. Построить `E_M^ext` и выбрать `A_M^F` по профилю `P_sit`.
4. Посчитать `Delta_M`.
5. Построить `E_R` и `E_D`.
6. Выполнить композицию `E_D o E_R o E_M^ext`.
7. Посчитать `rho(x)`.
8. Выбрать безопасное действие.
9. Сохранить след: `gamma`, `I(E_G)`, `Delta`, `D_ij`, `reason`.

## 10. Соответствие коду

| Математический объект | Реализация |
|---|---|
| `u_ent`, `u_mar` | `fuzzyxai/risk/uncertainty.py` |
| `rho(x)` | `fuzzyxai/risk/risk_function.py` |
| `pi_obs` | `fuzzyxai/risk/decision_policy.py` |
| Обёртка над моделью | `fuzzyxai/risk/risk_aware_model.py` |
| Полный маршрут | `fuzzyxai/risk/observer_pipeline.py`, `full_observer_pipeline.py` |
| Отчёт | `reports/full_observer_pipeline/` |
