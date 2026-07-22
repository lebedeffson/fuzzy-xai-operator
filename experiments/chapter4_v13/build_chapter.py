from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from .common import ARTIFACTS, ROOT, read_json, sha256_file


SOURCE = ROOT / "dissertation_artifacts" / "chapter4_v13" / "source_v12" / "Глава_4_FuzzyXAI_финальная_сдаваемая_редакция_v12.docx"
FINAL = ROOT / "dissertation_artifacts" / "chapter4_v13" / "final"
DOCX = FINAL / "Глава_4_FuzzyXAI_эмпирическая_редакция_v13.docx"
PDF = FINAL / "Глава_4_FuzzyXAI_эмпирическая_редакция_v13.pdf"
CHANGELOG = FINAL / "Глава_4_FuzzyXAI_v13_changelog.md"
VALIDATION = FINAL / "Глава_4_FuzzyXAI_v13_validation_report.md"


def _run(*command: str, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _format(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return "н/д"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, int):
        return f"{value:,}".replace(",", " ")
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}".replace(",", " ")
        return f"{value:.{digits}f}".replace(".", ",")
    return str(value).replace("_", " ")


def _markdown_table(frame: pd.DataFrame, labels: dict[str, str], digits: int = 4) -> str:
    columns = list(labels)
    header = "| " + " | ".join(labels[column] for column in columns) + " |"
    separator = "| " + " | ".join("---:" if pd.api.types.is_numeric_dtype(frame[column]) else ":---" for column in columns) + " |"
    rows = []
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(_format(row[column], digits) for column in columns) + " |")
    return "\n".join((header, separator, *rows))


def _replace_section(text: str, heading: str, next_heading: str, replacement: str) -> str:
    pattern = rf"(?ms)^## {re.escape(heading)}.*?(?=^## {re.escape(next_heading)})"
    updated, count = re.subn(pattern, replacement.rstrip() + "\n\n", text, count=1)
    if count != 1:
        raise RuntimeError(f"could not replace section {heading}")
    return updated


def _safe_human_review_section() -> str:
    return """## 4.14 Воспроизводимость и редакционная апробация формы объяснений

Воспроизводимость вычислительных результатов обеспечивается фиксацией версий кода, конфигураций, входных идентификаторов, случайных состояний и контрольных сумм артефактов. Числовые результаты главы связаны с машинно-читаемой картой доказательств; подробные пути, статусы и SHA256 вынесены в приложение.

Материалы проходили качественную редакционную апробацию специалистами, знакомыми с тематикой объяснимого искусственного интеллекта. Замечания использовались для уточнения терминологии, отделения вычислительного факта от интерпретации и более явного описания ограничений. Индивидуальные анкеты, количественный протокол, контрольная группа и статистика согласия не сохранялись. Поэтому апробация не используется для подтверждения гипотез о понятности, правильности действий или предметной безопасности.

Для будущего количественного исследования подготовлен отдельный протокол с критериями включения, рандомизацией, базовым условием, заданиями на понимание ограничений и планом статистического анализа. В настоящей главе он рассматривается только как направление дальнейшей работы.
"""


def _policy_conclusion(summary: dict[str, object]) -> str:
    primary = summary["primary_comparison"]
    positive = (
        float(primary["absolute_rate_reduction"]) > 0
        and float(primary["ci_lower"]) > 0
        and float(primary["holm_adjusted_p"]) < 0.05
    )
    if positive:
        return (
            "В заранее зафиксированном первичном сравнении полный контроллер показал статистически подтверждённое "
            "снижение числа ошибочных автоматических действий относительно выбранной на validation-части простой политики. "
            "Вывод относится только к данному контуру и не меняет отрицательный статус H3-original."
        )
    return (
        "В заранее зафиксированном первичном сравнении полный контроллер не показал статистически подтверждённого "
        "преимущества над простой политикой, выбранной только на validation-части. Этот результат сохранён без подбора "
        "порога после открытия меток и согласуется с отрицательным статусом H3-original."
    )


def _new_section() -> str:
    dataset = read_json(ARTIFACTS / "manifests" / "dataset_manifest.json")
    explanations = read_json(ARTIFACTS / "explanations" / "sealed_test_summary.json")
    policy_summary = read_json(ARTIFACTS / "policies" / "summary.json")
    evidence = read_json(ARTIFACTS / "evidence_map.json")
    manifest = read_json(ARTIFACTS / "runtime" / "manifest.json")
    case = read_json(ARTIFACTS / "end_to_end_case" / "action.json")
    case_timing = read_json(ARTIFACTS / "end_to_end_case" / "stage_timings.json")

    contour = pd.read_csv(ARTIFACTS / "tables" / "modern_contour.csv")
    policies = pd.read_csv(ARTIFACTS / "tables" / "policies_budget_20.csv")
    route = pd.read_csv(ARTIFACTS / "tables" / "route_validator.csv")
    runtime = pd.read_csv(ARTIFACTS / "tables" / "end_to_end_runtime.csv")
    hypotheses = pd.read_csv(ARTIFACTS / "tables" / "hypothesis_status.csv")

    policy_names = {
        "always_accept": "Безусловное принятие",
        "max_confidence": "Порог уверенности",
        "calibrated_confidence": "Калиброванная уверенность",
        "predictive_entropy": "Предиктивная энтропия",
        "weighted_linear": "Взвешенная линейная",
        "explainer_disagreement": "Расхождение объяснителей",
        "simple_or": "Логическое OR",
        "provenance_only": "Только происхождение",
        "predictive_risk_P0": "Предиктивный риск P0",
        "full_fuzzyxai_P1": "Риск P1",
        "full_fuzzyxai": "Полный FuzzyXAI",
        "random_matched_budget": "Случайная проверка",
    }
    policies["policy"] = policies["policy"].map(policy_names)
    policies = policies.sort_values("wrong_automatic_actions")

    route_names = {
        "simple_or": "Логическое OR",
        "independent_if_else": "Независимые if-else",
        "weighted_fault_score": "Взвешенная оценка",
        "typed_route_validator": "Типизированный валидатор",
    }
    group_names = {
        "clean": "Корректный маршрут",
        "registered_single": "Одиночные нарушения",
        "registered_compositional": "Композиционные нарушения",
        "held_out_fault_types": "Отложенные типы",
    }
    route["method"] = route["method"].map(route_names)
    route["group"] = route["group"].map(group_names)

    hypothesis_names = {
        "not_supported": "не подтверждена",
        "supported": "подтверждена в указанной области",
        "positive_effect": "положительный эффект в данном контуре",
        "no_confirmatory_advantage": "подтверждающее преимущество не установлено",
        "measured": "измерена в зарегистрированном контуре",
        "supported_registered_library": "подтверждена для зарегистрированной библиотеки",
        "descriptive": "описательное измерение",
        "exploratory": "разведочный результат",
    }
    hypotheses["result"] = hypotheses["result"].map(hypothesis_names).fillna(hypotheses["result"])
    hypotheses["scope"] = hypotheses["scope"].str.replace("frozen v1.3.0", "зафиксированный выпуск v1.3.0", regex=False)
    hypotheses["scope"] = hypotheses["scope"].str.replace("matched coverage sealed test", "сопоставимое покрытие, изолированная выборка", regex=False)
    hypotheses["scope"] = hypotheses["scope"].str.replace("registered single and compositional route faults", "зарегистрированные одиночные и композиционные нарушения", regex=False)
    hypotheses["scope"] = hypotheses["scope"].str.replace("end-to-end N=1..10000", "полный конвейер, N=1...10 000", regex=False)
    hypotheses["scope"] = hypotheses["scope"].str.replace("held-out fault types", "отложенные типы нарушений", regex=False)

    figures = ARTIFACTS / "figures"
    deviation = dataset["protocol_deviation"]
    primary = policy_summary["primary_comparison"]
    runtime_env = manifest["environment"]
    split_rows = dataset["processed_files"]
    actual_total = sum(int(item["rows"]) for item in split_rows.values())

    return f"""## 4.26 Современный прикладной контур и полная вычислительная стоимость

Для проверки переносимости на современную предварительно обученную модель дополнительно выполнен зафиксированный до расчёта результатов текстовый контур AG News с DistilBERT. Модель и ревизии источников не переобучались и не заменялись после просмотра результатов. Предсказания получены для {_format(actual_total)} допущенных объектов, а два локальных объяснения построены для {_format(int(explanations['objects']))} стратифицированно выбранных объектов из изолированной выборки. Числа этого раздела связаны с {_format(len(evidence['entries']))} числовыми записями карты доказательств.

### 4.26.1 Данные, модель и контроль утечек

До запуска модели аудит обнаружил повторы в закреплённой ревизии исходного набора. Официальная test-часть была сохранена, а {_format(int(deviation['excluded_training_rows']))} повторяющихся или совпадающих с test обучающих строк исключены до модельных вычислений. После исключения размеры частей составили {_format(int(split_rows['train']['rows']))}, {_format(int(split_rows['validation']['rows']))} и {_format(int(split_rows['sealed_test']['rows']))} объектов. Пересечения идентификаторов и нормализованных текстов между частями равны нулю. Это зарегистрированное отклонение не изменило модель, политики, показатели или критерии успеха.

{_markdown_table(contour, {'dataset': 'Набор', 'objects_total': 'Всего объектов', 'train': 'Обучение', 'validation': 'Настройка', 'sealed_test': 'Изолированная выборка', 'test_accuracy': 'Точность модели', 'local_explanations': 'Объяснения', 'canonical_hash_preservation': 'Сохранение хэша'})}

Таблица 4.37 — Современный прикладной контур AG News

### 4.26.2 Локальные объяснения и каноническое сохранение

Для каждого выбранного объекта рассчитаны Integrated Gradients и маскирование токенов. Сохраняемый канонический объект включает идентичность токена, знак, величину, ранг, параметры объяснителя, ревизию модели и контрольную сумму. Доля точного сохранения канонического хэша составила {_format(float(explanations['canonical_hash_preservation_rate']))}. Значения верности удаления, устойчивости к малым возмущениям и согласия двух методов приведены как измерения данного контура, а не как доказательство универсального превосходства.

<img src="{(figures / 'explanation_quality.png').resolve()}" style="width:15cm" />

Рисунок 4.25 — Распределение показателей локального объяснения в современном текстовом контуре

### 4.26.3 Политики при сопоставимом покрытии

Все политики направляют на проверку одинаковую долю объектов для каждого бюджета. Основной бюджет равен 20 %, а лучшая простая политика выбрана только на validation-части. Контрольные метки изолированной части открывались исключительно после записи и хэширования оценок риска и действий политик. {_policy_conclusion(policy_summary)}

Первичный scoring-пакет был признан недействительным при послерезультатном техническом аудите: контрольная политика безусловного принятия ошибочно получила общий бюджет проверки, а конечная bootstrap-оценка могла быть записана как p = 0. Исходный пакет сохранён с invalid marker. Затем выполнено только повторное вычисление метрик по неизменённым прогнозам, score rows, порогам и выбранной baseline-политике; модели и действия остальных политик не менялись. Поэтому итог трактуется как scoring-only recovery с зарегистрированным протокольным отклонением.

{_markdown_table(policies, {'policy': 'Политика', 'automatic_coverage': 'Покрытие', 'wrong_automatic_actions': 'Ошибочные действия', 'selective_risk': 'Риск', 'manual_review_load': 'На проверке', 'false_blocks': 'Ложные блокировки', 'total_cost': 'Стоимость'})}

Таблица 4.38 — Сопоставление политик при бюджете проверки 20 %

Первичная абсолютная разность долей ошибочных автоматических действий составляет {_format(float(primary['absolute_rate_reduction']), 6)}, 95%-й доверительный интервал [{_format(float(primary['ci_lower']), 6)}; {_format(float(primary['ci_upper']), 6)}], скорректированное значение p = {_format(float(primary['holm_adjusted_p']), 6)}. Знак эффекта трактуется относительно простой политики, заранее выбранной на validation-части.

<img src="{(figures / 'policy_risk_coverage.png').resolve()}" style="width:15cm" />

Рисунок 4.26 — Риск и автоматическое покрытие политик при одинаковых бюджетах проверки

### 4.26.4 Типизированная проверка маршрута

Независимое сравнение охватывает корректные маршруты, одиночные нарушения, их композиции и типы, не участвовавшие в настройке простых базовых схем. Типизированный валидатор возвращает не только общий запрет, но также тип и компонент нарушения. Результат относится к зарегистрированной библиотеке и не доказывает обнаружение произвольного неизвестного сбоя.

{_markdown_table(route, {'group': 'Группа', 'method': 'Метод', 'n': 'N', 'precision': 'Точность', 'recall': 'Полнота', 'f1': 'F1', 'false_certification': 'Ложная сертификация', 'component_localization_accuracy': 'Локализация'})}

Таблица 4.39 — Диагностика одиночных, композиционных и отложенных нарушений

<img src="{(figures / 'route_faults.png').resolve()}" style="width:15cm" />

Рисунок 4.27 — Качество обнаружения нарушений для четырёх проверяющих схем

### 4.26.5 Полная стоимость конвейера

Время разложено на модель, внешний объяснитель, собственный слой FuzzyXAI и сериализацию. Выполнено по {int(manifest['repetitions'])} измерений после прогрева. Окружение: {runtime_env['processor']}; RAM {_format(int(runtime_env['ram_total_bytes']))} байт; {runtime_env['gpu']}; Python {runtime_env['python']}; PyTorch {runtime_env['packages']['torch']}; доступно {int(runtime_env['cpu_count'])} вычислительных потоков. GPU не была изолирована: параллельно выполнялся независимый пользовательский процесс, зафиксированный в environment snapshot. Поэтому абсолютные времена являются описательными для данного разделяемого окружения. Показатели прежнего опыта на 5 млн записей по-прежнему относятся только к кэшированному операторному слою и не используются как характеристика полного конвейера.

{_markdown_table(runtime, {'explainer': 'Объяснитель', 'n': 'N', 'repetitions': 'Повторы', 'model_seconds_median': 'Модель, с', 'explainer_seconds_median': 'Объяснитель, с', 'fuzzyxai_seconds_median': 'FuzzyXAI, с', 'serialization_seconds_median': 'Сериализация, с', 'total_seconds_median': 'Полное время, с'})}

Таблица 4.40 — Время этапов современного текстового контура

{_markdown_table(runtime, {'explainer': 'Объяснитель', 'n': 'N', 'objects_per_second_median': 'Объектов/с', 'peak_rss_bytes_median': 'Пиковая RAM, байт', 'peak_vram_bytes_median': 'Пиковая VRAM, байт', 'fuzzyxai_time_fraction': 'Доля FuzzyXAI', 'explainer_time_fraction': 'Доля объяснителя'})}

Таблица 4.41 — Производительность, память и доли времени

<img src="{(figures / 'runtime_decomposition.png').resolve()}" style="width:16cm" />

Рисунок 4.28 — Декомпозиция полного времени по этапам конвейера

### 4.26.6 Воспроизводимый сквозной объект

Отдельный объект выбран детерминированным правилом до просмотра его метки. Для него сохранены ссылка на вход, прогноз, локальные вклады, канонический артефакт, граф происхождения, диагностическое состояние, действие, пользовательская карточка, аудиторский вывод и время каждого этапа. Контроллер выбрал действие «{_format(case['action'])}», структурный статус «{_format(case['hard_guard_status'])}»; полное измеренное время воспроизведения составило {_format(float(case_timing['total_seconds']), 6)} с. Текст исходного объекта не включён в выпуск из-за неопределённого статуса лицензии upstream-карточки AG News.

| Артефакт | Проверяемое содержание |
|:---|:---|
| Ссылка на вход | Идентификатор, исходный split, индекс и SHA256 нормализованного текста |
| Локальное объяснение | Токены и значения Integrated Gradients и маскирования |
| Канонический объект | Неизменяемая полезная нагрузка и контрольная сумма |
| Граф происхождения | Переход от объекта и модели к объяснению и действию |
| Диагностика | Структурный статус, коды причин и отсутствующие свидетельства |
| Аудит | Trace ID, версии и контрольная сумма детерминированного повтора |

Таблица 4.42 — Состав воспроизводимого сквозного примера

### 4.26.7 Статус новых проверок

{_markdown_table(hypotheses, {'hypothesis': 'Проверка', 'result': 'Результат', 'scope': 'Область'})}

Таблица 4.43 — Статус проверок дополнительного контура v13

Новый контур расширяет эмпирическую базу современной моделью, крупным набором, сопоставимыми бюджетами и полным измерением времени. Он не меняет зафиксированные отрицательные результаты H3-original, H5-P-original и H6-general и не создаёт утверждений о пользовательской понятности или предметной безопасности.

"""


def _clean_claims(text: str) -> str:
    text = re.sub(
        r"После автоматизированной предварительной проверки пользовательские объяснения.*?не заменяет количественную\s+проверку эффективности решений\.",
        "Форма пользовательского представления дорабатывалась редакционно; количественная пользовательская проверка в исследовании не проводилась.",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"Количественная гипотеза H7-B об улучшении\s+сокращённой проекции не подтверждена, однако качественная оценка.*?формулировок\.",
        "Количественная гипотеза H7-B об улучшении сокращённой проекции не подтверждена.",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"Качественная пользовательская оценка\s+поддерживает понятность итоговых формулировок, тогда как предметная\s+эффективность и безопасность требуют отдельного прикладного\s+исследования\.",
        "Пользовательская понятность и предметная эффективность требуют отдельного количественного исследования.",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"Качественная\s+оценка с участием людей.*?не заявляются как доказанные свойства\.",
        "Редакционная апробация не используется как доказательство понятности, предметной безопасности или правильности рекомендуемого действия.",
        text,
        flags=re.S,
    )
    text = text.replace(
        "| Границы метода | H3, H5-P и H6-general не подтверждены; качественная пользовательская оценка поддерживает понятность текста, но не доказывает предметную безопасность или преимущество политики |",
        "| Границы метода | H3, H5-P и H6-general не подтверждены; количественная пользовательская и предметная проверка не проводилась |",
    )
    return text


def _renumber_labels(text: str, kind: str) -> str:
    pattern = re.compile(rf"{kind} 4\.(\d+[а-я]?)", flags=re.I)
    mapping: dict[str, int] = {}
    for match in pattern.finditer(text):
        key = match.group(1).lower()
        if key not in mapping:
            mapping[key] = len(mapping) + 1
    return pattern.sub(lambda match: f"{kind} 4.{mapping[match.group(1).lower()]}", text)


def build() -> dict[str, object]:
    required = (
        ARTIFACTS / "manifests" / "validation.json",
        ARTIFACTS / "evidence_map.json",
        ARTIFACTS / "tables" / "modern_contour.csv",
        ARTIFACTS / "figures" / "runtime_decomposition.png",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"chapter build requires frozen evidence: {missing}")
    validation = read_json(ARTIFACTS / "manifests" / "validation.json")
    if not validation.get("passed"):
        raise RuntimeError("chapter build requires passing evidence validation")

    FINAL.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fuzzyxai-v13-chapter-") as temporary:
        work = Path(temporary)
        markdown = work / "chapter.md"
        _run("pandoc", str(SOURCE), "-t", "gfm", "--extract-media", str(work), "-o", str(markdown), cwd=work)
        text = markdown.read_text(encoding="utf-8")
        text = re.sub(
            r"^# .*?$",
            "# Глава 4. Практическая реализация и экспериментальная проверка структурной состоятельности объяснительно-аудиторского слоя FuzzyXAI",
            text,
            count=1,
            flags=re.M,
        )
        text = _replace_section(text, "4.14 Воспроизводимость и качественная оценка объяснений", "4.15 Независимый подтверждающий контур и доказательная линия v1.3.0", _safe_human_review_section())
        text = _clean_claims(text)
        marker = "## 4.26 Обсуждение результатов и выводы по главе"
        if marker not in text:
            raise RuntimeError("v12 conclusion marker is missing")
        text = text.replace(marker, _new_section() + "\n## 4.27 Обсуждение результатов и выводы по главе", 1)
        text = text.replace(
            "Масштабируемость операторного слоя проверена до пяти миллионов записей,\nоднако стоимость исходного объяснителя исключена.",
            "Масштабируемость кэшированного операторного слоя проверена отдельно до пяти миллионов записей. Дополнительный контур v13 измеряет полное время модели, объяснителя, FuzzyXAI и сериализации на доступных размерах; эти два результата не смешиваются.",
        )
        text = text.replace(
            "Практическая реализация представлена публичным API, адаптерами четырёх\nмодальностей, типизированным OperatorRoute, проверяемым ProofTrace и\nэкспортом доказательного пакета.",
            "Практическая реализация представлена публичным API, адаптерами четырёх модальностей, типизированным маршрутом, проверяемым доказательным следом и экспортом доказательного пакета. Дополнительный контур с DistilBERT и AG News подтверждает техническую переносимость на современную предварительно обученную модель и раскрывает полную вычислительную стоимость.",
        )
        text = _renumber_labels(text, "Таблица")
        text = _renumber_labels(text, "Рисунок")
        markdown.write_text(text, encoding="utf-8")
        _run("pandoc", str(markdown), "--reference-doc", str(SOURCE), "-o", str(DOCX), cwd=work)

    pdf_dir = FINAL / "pdf_render"
    if pdf_dir.exists():
        shutil.rmtree(pdf_dir)
    pdf_dir.mkdir(parents=True)
    _run("libreoffice", "--headless", "--convert-to", "pdf", "--outdir", str(pdf_dir), str(DOCX))
    rendered = pdf_dir / f"{DOCX.stem}.pdf"
    if not rendered.exists():
        raise RuntimeError("LibreOffice did not produce the PDF")
    shutil.move(rendered, PDF)
    shutil.rmtree(pdf_dir)

    shutil.copy2(ARTIFACTS / "evidence_map.json", FINAL / "Глава_4_FuzzyXAI_v13_evidence_map.json")
    shutil.copy2(ARTIFACTS / "leakage_audit.json", FINAL / "Глава_4_FuzzyXAI_v13_leakage_audit.json")
    shutil.copy2(ARTIFACTS / "validation_report.md", FINAL / "Глава_4_FuzzyXAI_v13_validation_report.md")
    CHANGELOG.write_text(
        "# Changelog главы 4 v13\n\n"
        "- Добавлен современный контур AG News с frozen DistilBERT, Integrated Gradients и маскированием токенов.\n"
        "- Добавлено сопоставление 12 политик при одинаковых бюджетах проверки.\n"
        "- Добавлена независимая проверка типизированного валидатора на одиночных, композиционных и отложенных нарушениях.\n"
        "- Полное время разделено на модель, объяснитель, FuzzyXAI и сериализацию.\n"
        "- Добавлен воспроизводимый сквозной объект и leakage audit.\n"
        "- H3-original, H5-P-original и H6-general оставлены отрицательными.\n"
        "- Удалены научные заявления о доказанной пользовательской понятности и предметной безопасности.\n"
        "- Показатель 5 млн объектов сохранён только для кэшированного операторного слоя.\n"
        "- AG News не включён в архив из-за неизвестного статуса лицензии в upstream-карточке.\n",
        encoding="utf-8",
    )
    return {"docx": DOCX, "pdf": PDF, "docx_sha256": sha256_file(DOCX), "pdf_sha256": sha256_file(PDF)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = build()
    print(f"PASS: chapter docx={result['docx_sha256']} pdf={result['pdf_sha256']}")


if __name__ == "__main__":
    main()
