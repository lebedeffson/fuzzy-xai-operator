# Инвентаризация реализации vNext

## Что уже закрыто

| Блок | Статус | Где проверяется |
|---|---|---|
| Глава 2: `E_k`, `Omega`, композиция, `D_ij` | готово | `tests/test_composition.py`, `proofs/chapter2_operator_proof.py` |
| `compose()` с `gamma_max` | готово | `tests/test_composition_well_defined.py` |
| `d_E`, `L(E)`, `I(E)` | готово | `tests/test_semantic_disagreement_pseudometric.py`, `tests/test_chain_loss_bound.py` |
| Глава 3: `F0`, `FI`, `FH`, `FNsrc`, `FML` | готово | `tests/test_reductions.py` |
| Редукции и `Delta` | готово | `tests/test_reduction_approximation_theorem.py` |
| Парето-выбор `A_M^F` | готово | `tests/test_operational_coverage_minimality.py` |
| Risk-Aware Observer | готово | `tests/test_risk_aware_model.py`, `tests/test_observer_no_cyclic_dependency.py` |
| Dataset observer pipeline | готово | `tests/test_dataset_observer_pipeline.py` |
| Выбор `A_M^F` в dataset pipeline | готово | `tests/test_dataset_observer_representation_integration.py` |
| Breast Cancer Wisconsin по прямой ссылке | готово | `examples/dataset_observer_demo.py --url ...breast_cancer_data.csv` |
| `E_R` и `E_A` в observer report | готово | `tests/test_dataset_observer_representation_integration.py` |
| Formal proof report | готово | `proofs/formal_theorem_checks.py` |
| Категориально-гомотопическое приложение | готово | `make category-hott-test`, `reports/category_hott/category_hott_checks.md` |
| Предпучки `Risk/Audit/User/Trace`, `AutoAccept`, Йонеда | готово | `tests/test_subpresheaf.py`, `tests/test_yoneda.py`, `tests/test_risk_context_acceptance.py` |
| Эксперименты главы 5 | готово | `make chapter5-experiments`, `reports/chapter5/chapter5_experiments.md` |
| Калиброванные веса observer + demo + LaTeX | готово | `make chapter5-demo`, `make chapter5-latex`, `tests/test_calibrated_observer_report_consistency.py` |

## Что осталось как следующий слой

1. Реальные скачанные CIT-датасеты: сейчас поддержаны локальный файл, GitHub/raw CSV и прямая ссылка, но не выбран конкретный большой набор для репозитория.
2. Отдельный PyTorch-demo: нужен без обязательной зависимости `torch` в базовом `requirements.txt`, лучше через optional example.
3. GUI-экран Dataset Observer: backend и отчёты готовы, интерфейс можно усилить отдельным проходом.

## Текущая главная линия

```text
Dataset -> P_data -> M(x) -> E_M^ext -> A_M^F -> I_pre -> rho(x) -> E_R -> E_A -> action/report
```

Это уже закрывает связку:

```text
глава 2: композиция объяснений
глава 3: тип неопределённости
observer: риск применения прогноза
appendix: Expl -> Set^{Expl^op} -> Path_Expl -> Rupture
```
