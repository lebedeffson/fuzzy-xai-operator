# Практическая реализация (главы 2–3)

## Соответствие диссертации и кода

| Раздел диссертации | Объект | Реализация | Проверка |
|---|---|---|---|
| 2 | `E_k` | `fuzzyxai/core/explanation_object.py` | `tests/test_composition.py` |
| 2 | `Omega_k` | `fuzzyxai/core/system_operator.py` | `apps/layered_demo.py` |
| 2 | `T_ij`, `D_ij` | `fuzzyxai/core/composition.py`, `fuzzyxai/core/diagnostics.py` | `tests/test_finite_category_laws.py`, `tests/test_hott_path_rupture.py` |
| 3 | `F_int`, `F_H`, `F_N_src` | `fuzzyxai/hierarchy/*`, `fuzzyxai/risk/representation_selection.py` | `tests/test_real_reduction_example.py` |
| 3 | `Expl_ext` | `fuzzyxai/category/expl_category.py` | `tests/test_finite_category_laws.py` |
| 3 | `Omega`, `chi_Auto` | `fuzzyxai/category/context_topos.py`, `fuzzyxai/category/subpresheaf.py` | `tests/test_subobject_classifier.py`, `tests/test_characteristic_morphism_auto.py` |
| 3 | `rho`, `action`, `chi_R`, `chi_R^crit` | `fuzzyxai/risk/risk_function.py`, `apps/services/layered_case.py`, `apps/layered_demo.py` | `tests/test_dataset_benchmark.py` |
| 3 | dataset modes | `fuzzyxai/datasets/registry.py`, `examples/check_dataset_modes.py` | `tests/test_dataset_modes_check.py` |

## Команды воспроизведения

```bash
PYTHONPATH=. pytest
make dataset-modes-check
make benchmark-dataset DATASET=breast_cancer
make real-reduction-example
make dissertation-demo-summary
make layered-demo PORT=8096
```

## Артефакты отчётов

- `reports/datasets/<dataset>/summary.json`
- `reports/datasets/<dataset>/summary.md`
- `reports/datasets/<dataset>/predictions.csv`
- `reports/real_reduction_example/breast_cancer_case.json`
- `reports/real_reduction_example/breast_cancer_case.md`
- `reports/dissertation_demo_summary.json`
- `reports/dissertation_demo_summary.md`
