# Категориально-топосная реализация (RU)

## Что реализовано

- Конечная категория объяснений `Expl_ext`: `fuzzyxai/category/expl_category.py`
- Предпучки контекстов: `fuzzyxai/category/presheaf.py`
- Контексты `RiskContext`, `AuditContext`, `UserContext`, `TraceContext`: `fuzzyxai/category/context_topos.py`
- Подпредпучок `AutoAccept` и проверка `chi_Auto`: `fuzzyxai/category/subpresheaf.py`

## Вычислимость для конечной категории

Для конечной `Expl_ext` все `Hom`-множества конечны, поэтому сита и подпредпучки можно перечислять явно.
Практически это реализовано через конечные множества действий в контекстах и фильтрацию (`Subpresheaf`), что эквивалентно конечной табличной реализации `Omega(E)`/`chi_Auto`.

## Проверки

```bash
PYTHONPATH=. pytest tests/test_finite_category_laws.py tests/test_subobject_classifier.py tests/test_characteristic_morphism_auto.py tests/test_hott_path_rupture.py -q
```

## Примечание для защиты

`chi_R(x)` и `chi_R^crit(x)` разведены:
- `chi_R(x)=1`: есть любой диагностический разрыв;
- `chi_R^crit(x)=1`: критический разрыв, авто-действие блокируется.
