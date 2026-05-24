# Категориально-гомотопическое расширение наблюдающего контура FuzzyXAI

Версия: `vNext Category/HoTT Extension`.
Назначение: приложение к главам 2-3 или отдельный раздел после главы 3.

Этот слой не заменяет системный оператор, иерархию представлений и Risk-Aware Observer. Он даёт безопасную математическую интерпретацию уже построенного контура. Главное ограничение: мы не утверждаем, что сама категория объяснений является топосом. Вместо этого строится стандартная конструкция:

\[
\mathsf{Expl}\rightarrow \widehat{\mathsf{Expl}}=\mathbf{Set}^{\mathsf{Expl}^{op}}\rightarrow \mathsf{Path}_{Expl}\rightarrow \mathsf{Rupture}.
\]

## 1. Назначение слоя

Глава 2 задаёт объяснительный объект, системный оператор, композицию, рассогласование и диагностику. Глава 3 заменяет обычную функцию принадлежности на выбранное представление неопределённости `A_k^F`. Категориально-гомотопический слой добавляет язык более высокого уровня:

| Уровень | Объект | Смысл |
|---|---|---|
| Категориальный | `Expl` | успешные согласования как морфизмы |
| Топосный | `Set^{Expl^op}` | контексты применимости как предпучки |
| HoTT | `Path_Expl`, `Rupture` | согласование как путь, диагностика как разрыв |

## 2. Исходные объекты глав 2-3

Базовый объект главы 2:

\[
E_k=\langle L_k,\mu_k,R_k,\alpha_k,u_k,\tau_k\rangle.
\]

Расширенный объект главы 3:

\[
E_k^{ext}=\langle L_k,A_k^{\mathcal F},R_k,\alpha_k,u_k,\tau_k,\Delta_k\rangle.
\]

Допустимое согласование должно удовлетворять:

\[
d_E^{ext}(T_{ij}(E_i),E_j)\leq\gamma_{max}.
\]

Если условие нарушено, возникает `D_ij`.

## 3. Категория допустимых объяснений `Expl`

Объекты:

\[
Obj(\mathsf{Expl})=\{E^{ext}: E^{ext}\text{ валиден относительно ExplainPlan}\}.
\]

Морфизмы:

\[
Hom_{\mathsf{Expl}}(E_i,E_j)=\{T_{ij}: d_E^{ext}(T_{ij}(E_i),E_j)\leq\gamma_{max}\}.
\]

`D_ij` не является морфизмом. Оно означает отсутствие допустимого перехода.

**Теорема 4.1. Категория допустимых объяснений.** Если множество валидных объектов и допустимых согласований задано внутри фиксированного `ExplainPlan`, тождества существуют, а допустимые согласования замкнуты относительно композиции, то `Expl` является малой категорией.

Доказательство. Тождественный морфизм задаётся самосогласованием объекта. Композиция задаётся последовательным применением согласований. Ассоциативность следует из ассоциативности композиции отображений на полях объекта. Малость следует из фиксации вычислительного контура. ∎

Код: `fuzzyxai/category/expl_category.py`, `fuzzyxai/category/morphism.py`.

## 4. Диагностическое расширение

Попытка согласования является частичным вычислением:

\[
T_{ij}:E_i\to E_j\sqcup\mathfrak D.
\]

**Теорема 4.2. Диагностика как граница категории.** Если для пары `E_i,E_j` нет допустимого `T_ij`, то `Hom_Expl(E_i,E_j)=empty`, а попытка согласования возвращает `D_ij`.

Код: `fuzzyxai/category/diagnostic_completion.py`.

## 5. Предпучковый топос контекстов

Над малой категорией объяснений строится:

\[
\widehat{\mathsf{Expl}}=\mathbf{Set}^{\mathsf{Expl}^{op}}.
\]

Предпучок:

\[
\mathcal P:\mathsf{Expl}^{op}\to\mathbf{Set}.
\]

Интерпретация:

\[
\mathcal P(E)=\{\text{контексты применимости }E\}.
\]

Для морфизма `f:E_i -> E_j`:

\[
\mathcal P(f):\mathcal P(E_j)\to\mathcal P(E_i).
\]

**Теорема 4.3. Существование топоса контекстов.** Если `Expl` малая категория, то `Set^{Expl^op}` является предпучковым топосом. Это стандартный факт теории категорий.

Код: `fuzzyxai/category/presheaf.py`, `fuzzyxai/category/context_topos.py`.

В реализации добавлены конкретные предпучки контекстов:

| Предпучок | Код | Значение |
|---|---|---|
| `RiskContext` | `fuzzyxai/category/context_topos.py` | допустимые действия наблюдателя |
| `AuditContext` | `fuzzyxai/category/context_topos.py` | уровни аудита и trace |
| `UserContext` | `fuzzyxai/category/context_topos.py` | пользовательские режимы представления |
| `TraceContext` | `fuzzyxai/category/context_topos.py` | требования к трассировке |

Подпредпучок `AutoAccept` реализован как `Subpresheaf`:

\[
AutoAccept(E)\subseteq RiskContext(E).
\]

Он содержит только действия `accept` и `lower_confidence`. Поэтому проверка автоматического применения становится контекстной: прогноз можно применять автоматически только если `AutoAccept(E)` непуст.

Для аудита добавлен представимый предпучок Йонеды:

\[
y(E)=Hom_{\mathsf{Expl}}(-,E).
\]

Он задаёт множество допустимых объяснительных предысторий, ведущих к объекту `E`.

## 6. Контекстная логика

Топосный слой не заменяет нечёткую логику. Он описывает, где объяснение применимо:

| Предикат | Смысл |
|---|---|
| `AutoAccept(E)` | объяснение допустимо для автоматического применения |
| `AuditReady(E)` | объяснение содержит достаточный trace |
| `UserReadable(E)` | объяснение можно показать пользователю после допустимой редукции |
| `Stable(E_t,E_{t+1})` | объяснение стабильно между версиями |

Один объект может быть `AuditReady=True`, но `AutoAccept=False`.

## 7. HoTT-интерпретация путей

Вводится направленный тип пути:

\[
\mathsf{Path}_{Expl}(E_i,E_j).
\]

Элемент:

\[
p=\langle T_{ij},\gamma_{ij},\Delta_{ij},\tau_{ij}\rangle.
\]

**Теорема 4.4. Путь и морфизм совпадают по существованию.** `Path_Expl(E_i,E_j)` населён тогда и только тогда, когда существует допустимый морфизм `E_i -> E_j`.

Код: `fuzzyxai/hott/path_type.py`.

## 8. Разрыв как тип

Если путь отсутствует, фиксируется:

\[
\mathsf{Rupture}(E_i,E_j).
\]

**Теорема 4.5. Диагностика как тип разрыва.** Если все кандидаты `T_ij` нарушают `ExplainPlan` или `gamma_max`, то `Path_Expl(E_i,E_j)=empty`, а `Rupture(E_i,E_j)` населён диагностикой `D_ij`.

Код: `fuzzyxai/hott/rupture_type.py`.

## 9. Динамический слой: дрейф объяснений

Для версий модели:

\[
E(t_0),E(t_1),...,E(t_n).
\]

Если все соседние переходы допустимы, строится составной динамический путь. Если хотя бы один переход превышает порог, фиксируется drift-rupture.

**Теорема 4.6. Составной путь дрейфа.** Если для всех соседних моментов `d_E(E(t_k),E(t_{k+1})) <= gamma_max`, то существует путь `E(t_0) -> E(t_n)`. Иначе цепочка разрывается.

Код: `fuzzyxai/hott/drift_path.py`.

## 10. Связь с Risk-Aware Observer

Основной маршрут наблюдателя:

\[
D\to P_{data}\to M(x)\to E_M^{ext}\to E_{pre}\to\rho(x)\to a^*(x)\to E_G.
\]

Категориальная интерпретация:

\[
E_M^{ext}\to E_R\to E_A.
\]

Если морфизмы существуют, строится путь `model -> risk -> action`. Если возникает `Rupture`, риск автоматического применения получает диагностический штраф:

\[
\rho(x)=w_p\rho_p+w_u u_M+w_I(1-I_{pre})+w_\Delta\Delta_M+w_D1_{D\neq empty}.
\]

Можно эквивалентно записать:

\[
\chi_R=1_{\mathsf{Rupture}\text{ inhabited}}.
\]

## 11. Алгоритмическая спецификация

| Модуль | Назначение |
|---|---|
| `fuzzyxai/category/expl_category.py` | объекты, морфизмы, identity, compose |
| `fuzzyxai/category/diagnostic_completion.py` | морфизм или диагностика |
| `fuzzyxai/category/presheaf.py` | предпучки и restriction maps |
| `fuzzyxai/category/context_topos.py` | дескриптор `Set^{Expl^op}` |
| `fuzzyxai/hott/path_type.py` | `ExplanationPath` |
| `fuzzyxai/hott/rupture_type.py` | `RuptureType` |
| `fuzzyxai/hott/drift_path.py` | временные пути |
| `fuzzyxai/hott/path_certificates.py` | JSON-сертификаты путей/разрывов |

## 12. Тесты и команды

```bash
make category-hott-test
PYTHONPATH=. python proofs/category_hott_checks.py
```

Проверяются:

- identity и ассоциативность;
- диагностическая граница;
- `P(id)=id`;
- `P(g o f)=P(f) o P(g)`;
- путь существует при морфизме;
- `RuptureType` создаётся из `D_ij`;
- temporal drift path.

Отчёты:

```text
reports/category_hott/category_hott_checks.json
reports/category_hott/category_hott_checks.md
```

## 13. Пример: успешное согласование

\[
E_M^{ext}\xrightarrow{\gamma=0.18}E_R\xrightarrow{\gamma=0.12}E_A.
\]

Существуют два морфизма и составной путь `E_model -> E_action`.

## 14. Пример: разрыв интерфейса

Если `gamma > gamma_max`, то:

\[
Hom_{Expl}(E_M,E_A)=empty,
\quad
Path_{Expl}(E_M,E_A)=empty,
\quad
Rupture(E_M,E_A)\text{ inhabited}.
\]

В observer это ведёт к `block` или `defer_to_human`.

## 15. Пример: временной дрейф

Если новая версия модели сохраняет accuracy, но меняет термы/правила так, что `d_E` превышает порог, возникает `D_drift`. Это означает, что модель нельзя заменить без аудита объяснительного интерфейса.

## 16. Как вставить в диссертацию

Лучший вариант: приложение Б. В главе 3 достаточно ссылки:

> Более абстрактная категориально-гомотопическая интерпретация предложенного наблюдающего контура приведена в приложении Б.

## 17. Ограничения

| Не утверждается | Корректная формулировка |
|---|---|
| `Expl` сама топос | над `Expl` строится `Set^{Expl^op}` |
| `D_ij` морфизм | `D_ij` свидетельствует об отсутствии морфизма |
| HoTT доказывает всю модель | HoTT даёт язык путей и разрывов |
| топос повышает accuracy | он уточняет контекст применимости |
| дрейф всегда допустим | путь существует только при `d_E <= gamma_max` |

## 18. JSON-отчёт

```json
{
  "category_hott_report": {
    "objects": ["E_model", "E_risk", "E_action"],
    "morphisms": [
      {"source": "E_model", "target": "E_risk", "gamma": 0.18, "valid": true}
    ],
    "paths": [
      {"source": "E_model", "target": "E_action", "length": 2, "valid": true}
    ],
    "ruptures": [],
    "presheaf_contexts": {
      "RiskContext": ["lower_confidence", "request_more_data"],
      "AuditContext": ["full_trace", "hash_verified"]
    }
  }
}
```

## 19. Формулировка для защиты

> Основная часть диссертации даёт вычислимый нечёткий наблюдатель. В приложении показано, что успешные согласования можно рассматривать как морфизмы категории объяснений, контексты - как предпучки, а диагностические состояния - как разрывы путей согласования. Это задаёт строгий математический язык для дальнейшего развития аппарата.

## 20. Источники

1. Mac Lane S., Moerdijk I. Sheaves in Geometry and Logic. Springer, 1992.
2. Johnstone P. T. Sketches of an Elephant. Oxford University Press, 2002.
3. The Univalent Foundations Program. Homotopy Type Theory. arXiv:1308.0729.
4. Awodey S., Pelayo A., Warren M. A. Voevodsky's Univalence Axiom in HoTT. arXiv:1302.4731.
5. Gratzer D., Weinberger J., Buchholtz U. Directed Univalence in Simplicial HoTT. arXiv:2407.09146.
6. Poernomo I. DHoTT: A Temporal Extension of HoTT for Semantic Drift. arXiv:2506.09671.
7. Zadeh L. A. Fuzzy Sets. Information and Control, 1965.

## Приложение: аксиомы слоя

| Аксиома | Смысл |
|---|---|
| C1 | каждый валидный `E_ext` является объектом `Expl` |
| C2 | морфизмом является только допустимое `T_ij` |
| C3 | `id_E` существует для каждого объекта |
| C4 | композиция допустима только внутри порога `gamma_max` |
| C5 | недопустимое согласование возвращает `D_ij`, а не морфизм |
| C6 | путь существует тогда и только тогда, когда существует морфизм |
| C7 | `RuptureType` населён тогда, когда путь пуст и есть диагностика |
