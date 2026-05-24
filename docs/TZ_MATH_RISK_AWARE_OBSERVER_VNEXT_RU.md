# Техническое задание vNext

## Усиление математического аппарата глав 2-3 и Risk-Aware Observer

## 1. Цель

Цель vNext - связать диссертационные объекты, теоремы, код, тесты и отчёты в одну воспроизводимую конструкцию:

```text
Dataset -> P_data -> M(x) -> E_M^ext -> E_pre -> rho(x) -> a*(x) -> E_G
```

Система должна показывать не только объяснение прогноза, но и контролируемое применение прогноза.

## 2. Граница утверждений

Нельзя утверждать, что наблюдатель универсально улучшает любую модель. Корректно:

> Наблюдатель не изменяет параметры модели. Он работает как внешний риск-ориентированный контур, который оценивает допустимость автоматического применения прогноза по объяснимости, неопределённости, потере редукции и диагностическим состояниям.

Успех фиксируется через снижение оценочной стоимости:

\[
RiskReduction=C_{before}-C_{after}.
\]

## 3. Математика главы 2

Нужно поддержать три результата:

1. корректность частичной композиции объяснений;
2. условную псевдометрику `d_E` на согласованных объектах;
3. верхнюю оценку потери для конечной цепочки.

Документ: `docs/FORMAL_THEOREMS_CH2_CH3_RU.md`.

Проверки:

```text
tests/test_composition_well_defined.py
tests/test_semantic_disagreement_pseudometric.py
tests/test_chain_loss_bound.py
```

## 4. Математика главы 3

Нужно поддержать три результата:

1. операциональное покрытие и относительную минимальность иерархии;
2. аппроксимационную корректность редукции;
3. завершимость редукции многоуровневого представления.

Проверки:

```text
tests/test_operational_coverage_minimality.py
tests/test_reduction_approximation_theorem.py
tests/test_multilevel_reduction_termination.py
```

## 5. Risk-Aware Observer

Функция риска должна использовать `I_pre`, а не `I(E_G)`:

\[
\rho(x)=w_p\rho_p(x)+w_u u_M(x)+w_I(1-I_{pre})+w_\Delta\Delta_M+w_D1[D_{pre}\ne\varnothing].
\]

Финальный индекс `I_final=I(E_G)` идёт в отчёт и аудит, но не участвует в первичном вычислении `rho(x)`.

Проверки:

```text
tests/test_observer_no_cyclic_dependency.py
tests/test_expected_cost_optimality.py
tests/test_dataset_observer_pipeline_correctness.py
```

## 6. Proof-скрипт

Добавить:

```text
proofs/formal_theorem_checks.py
```

Он должен создавать:

```text
reports/formal_theorems/formal_theorem_checks.json
reports/formal_theorems/formal_theorem_checks.md
```

Команды:

```bash
make formal-proof
make proof
```

## 7. Dataset Observer

Поддержать режимы:

```bash
PYTHONPATH=. python examples/dataset_observer_demo.py --sample breast_cancer
PYTHONPATH=. python examples/dataset_observer_demo.py --file data/my_dataset.csv --target target
PYTHONPATH=. python examples/dataset_observer_demo.py --url "https://.../dataset.csv" --target target
```

## 8. GUI

В `defense_demo` маршрут должен быть представлен как:

```text
Dataset -> Profile -> Model -> Prediction -> E_M^ext -> I_pre -> rho -> Action -> E_G -> Diagnostics
```

## 9. Что не включать в основные результаты

Топосы, persistent homology и HoTT не включать как доказанные результаты текущей работы. Их можно оставить в перспективах.

## 10. Критерии готовности

```text
1. make test проходит.
2. make proof проходит и запускает formal_theorem_checks.py.
3. make dataset-observer проходит.
4. make full-observer проходит.
5. CHAPTER_MAPPING.md и README.md содержат ссылки на новые проверки.
```
