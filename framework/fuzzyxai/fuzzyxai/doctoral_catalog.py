"""Author's 47-method canonical XAI taxonomy from Appendix A.5, chapter 1.

The method IDs m01..m47 are STABLE table row identifiers. A noncanonical
implementation must use its own method ID rather than silently inheriting
classification from a different configuration.
"""
from __future__ import annotations

AUTHOR_METHODS: tuple[tuple[str, str, int, str, str, str], ...] = (
    ('m01', 'Линейные и логистические модели', 1958, 'intrinsic', 'features', 'auditor'),
    ('m02', 'Обобщённые аддитивные модели (GAM, EBM)', 1986, 'intrinsic', 'features', 'auditor'),
    ('m03', 'Скоринговые системы (RiskSLIM)', 2016, 'intrinsic', 'features', 'auditor'),
    ('m04', 'Рассуждение по прецедентам, kNN', 1994, 'intrinsic', 'data', 'end_user'),
    ('m05', 'Нечёткие системы на правилах', 1975, 'intrinsic', 'logic', 'auditor'),
    ('m06', 'Деревья решений', 1984, 'intrinsic', 'logic', 'auditor'),
    ('m07', 'ANFIS и нейро-нечёткие сети', 1993, 'intrinsic', 'logic', 'auditor'),
    ('m08', 'Списки и множества правил (CORELS, BRL)', 2015, 'intrinsic', 'logic', 'auditor'),
    ('m09', 'Байесовские сети', 1985, 'intrinsic', 'causal', 'researcher'),
    ('m10', 'Механизм внимания как объяснение', 2015, 'builtin', 'features', 'end_user'),
    ('m11', 'Самообъясняющиеся сети (SENN)', 2018, 'builtin', 'representation', 'end_user'),
    ('m12', 'Прототипные сети (ProtoPNet)', 2019, 'builtin', 'representation', 'end_user'),
    ('m13', 'Концептное узкое место (CBM)', 2020, 'builtin', 'representation', 'end_user'),
    ('m14', 'Глубокие k ближайших соседей (DkNN)', 2018, 'builtin', 'data', 'developer'),
    ('m15', 'Генерация с атрибуцией источников (RAG)', 2020, 'builtin', 'data', 'end_user'),
    ('m16', 'Обоснования на естественном языке (e-SNLI)', 2018, 'builtin', 'logic', 'end_user'),
    ('m17', 'Объяснение на графе знаний', 2018, 'builtin', 'logic', 'end_user'),
    ('m18', 'Нейро-символические сети логического вывода', 2020, 'builtin', 'logic', 'auditor'),
    ('m19', 'Цепочка рассуждений LLM (CoT)', 2022, 'builtin', 'logic', 'end_user'),
    ('m20', 'Обучение причинных представлений', 2021, 'builtin', 'causal', 'researcher'),
    ('m21', 'Перестановочная важность признаков', 2001, 'posthoc', 'features', 'auditor'),
    ('m22', 'PDP, ICE, ALE', 2001, 'posthoc', 'features', 'auditor'),
    ('m23', 'Карты значимости по градиенту', 2013, 'posthoc', 'features', 'developer'),
    ('m24', 'Окклюзия и возмущения (RISE)', 2014, 'posthoc', 'features', 'developer'),
    ('m25', 'LRP и DeepLIFT', 2015, 'posthoc', 'features', 'developer'),
    ('m26', 'LIME', 2016, 'posthoc', 'features', 'end_user'),
    ('m27', 'SHAP (значения Шепли)', 2017, 'posthoc', 'features', 'end_user'),
    ('m28', 'Контрфактические объяснения', 2017, 'posthoc', 'features', 'end_user'),
    ('m29', 'Grad-CAM и CAM', 2017, 'posthoc', 'features', 'end_user'),
    ('m30', 'Integrated Gradients', 2017, 'posthoc', 'features', 'developer'),
    ('m31', 'Алгоритмическое обжалование (recourse)', 2019, 'posthoc', 'features', 'end_user'),
    ('m32', 'Объяснения GNN (GNNExplainer)', 2019, 'posthoc', 'features', 'developer'),
    ('m33', 'Лингвистические резюме объяснений (нечёткие)', 2019, 'posthoc', 'features', 'end_user'),
    ('m34', 'Визуализация признаков и активаций', 2014, 'posthoc', 'representation', 'developer'),
    ('m35', 'Рассечение сети (Network Dissection)', 2017, 'posthoc', 'representation', 'researcher'),
    ('m36', 'Зондирующие классификаторы (probing)', 2017, 'posthoc', 'representation', 'developer'),
    ('m37', 'TCAV и концептные векторы', 2018, 'posthoc', 'representation', 'researcher'),
    ('m38', 'Функции влияния и TracIn', 2017, 'posthoc', 'data', 'developer'),
    ('m39', 'Документирование данных (Datasheets, Data statements)', 2018, 'posthoc', 'data', 'regulator'),
    ('m40', 'Data Shapley и оценка данных', 2019, 'posthoc', 'data', 'developer'),
    ('m41', 'Объяснение примерами и контрпримерами', 2019, 'posthoc', 'data', 'end_user'),
    ('m42', 'Глобальный суррогат и извлечение правил (TREPAN)', 1996, 'posthoc', 'logic', 'auditor'),
    ('m43', 'Аргументационные объяснения', 2017, 'posthoc', 'logic', 'end_user'),
    ('m44', 'Anchors', 2018, 'posthoc', 'logic', 'end_user'),
    ('m45', 'Документирование модели (Model Cards, FactSheets)', 2019, 'posthoc', 'logic', 'regulator'),
    ('m46', 'Механистическая интерпретируемость (схемы)', 2020, 'posthoc', 'logic', 'researcher'),
    ('m47', 'Причинные объяснения на структурной модели', 2019, 'posthoc', 'causal', 'researcher'),
)

def catalog_lookup(method_id: str) -> dict[str, str | int] | None:
    for key, label, year, source, object_type, audience in AUTHOR_METHODS:
        if key == method_id:
            return {"method_id": key, "label": label, "year": year,
                    "source": source, "object": object_type, "audience": audience}
    return None

def validate_catalog() -> dict[str, object]:
    assert len(AUTHOR_METHODS) == 47
    assert len({row[0] for row in AUTHOR_METHODS}) == 47
    assert all(row[3] in {"intrinsic", "builtin", "posthoc"}
               and row[4] in {"features", "logic", "data", "representation", "causal"}
               and row[5] in {"end_user", "regulator", "researcher", "developer", "auditor"}
               for row in AUTHOR_METHODS)
    from collections import Counter
    return {"count": len(AUTHOR_METHODS),
            "source_counts": dict(Counter(row[3] for row in AUTHOR_METHODS)),
            "distinct_cells": len({(row[3], row[4], row[5]) for row in AUTHOR_METHODS})}
