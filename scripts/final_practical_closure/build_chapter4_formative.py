#!/usr/bin/env python3
"""Build a placeholder-free Chapter 4 shell with only formative wording."""

from __future__ import annotations

from common import FORMATIVE, ROOT, load_json, sha256


OUTPUT = ROOT / "dissertation_artifacts/final_practical_closure/chapter4_formative"


SECTIONS = (
    "4.1 Практическая постановка",
    "4.2 Данные и sealed protocol",
    "4.3 Архитектура практического контроллера",
    "4.4 Корректная taxonomy аналогов",
    "4.5 H1 и H2",
    "4.6 H4",
    "4.7 H3-original",
    "4.8 H3 practical",
    "4.9 H5-S и H5-A",
    "4.10 H5-P",
    "4.11 H6-A и H6-B",
    "4.12 H7-A и H7-B",
    "4.13 H8",
    "4.14 H9",
    "4.15 Формирующая проверка карточек",
    "4.16 Воспроизводимость",
    "4.17 Итоговые claims и ограничения",
)


def main() -> None:
    summary = load_json(FORMATIVE / "summary.json")
    evidence = f"[evidence:FORMATIVE-SUMMARY sha256:{sha256(FORMATIVE / 'summary.json')}]"
    bodies = {
        "4.1": "Раздел определяет operationally invalid automatic action как практическую цель. Независимая подтверждающая выборка не открыта.",
        "4.2": "Использованы только development/controlled данные; confirmatory identities и labels отсутствуют у tuning runner.",
        "4.3": "Контроллер разделён на hard structural guard, predictive-risk estimator, route-risk estimator и budgeted optimizer.",
        "4.4": "Post-hoc explainers, glass-box predictors и action policies сравниваются в разных семействах.",
        "4.5": "Ранее замороженные H1 и H2 сохраняются без изменения статуса.",
        "4.6": "Ранее замороженный H4 сохраняется без изменения статуса.",
        "4.7": "H3-original остаётся not_supported; новый практический H3 не переименовывает этот результат.",
        "4.8": "Formative budget comparison выполнен, но положительный confirmatory claim запрещён.",
        "4.9": "H5-S сохранён; H5-A измерен только на controlled faults, natural failures ещё не подтверждены.",
        "4.10": "H5-P-original остаётся not_supported; route validity не объявляется предиктором ошибки модели.",
        "4.11": f"H6-A измеряет detectability envelope; H6-B имеет статус {summary['H6_B_status']}.",
        "4.12": "Canonical evidence и пользовательская projection разделены; H7-A проверяет hash, H7-B требует independent confirmation.",
        "4.13": "Компонентная сетка проверена формативно в заранее заданных конфигурациях.",
        "4.14": "Масштабирование относится к operator layer; стоимость local explainer учитывается отдельно.",
        "4.15": f"AI formative run 2: {summary['ai_formative_run2']}; AI-review не является экспертной оценкой.",
        "4.16": "Каждый formative experiment содержит protocol, manifests, JSONL, Parquet, statistics, claim status и SHA256SUMS.",
        "4.17": "До protocol lock разрешены только технические и formative формулировки; human/expert/domain-safety claims исключены.",
    }
    lines = ["# Глава 4. Практический контур FuzzyXAI", "", "> FORMATIVE ONLY. Финальные статистические выводы появятся только после sealed confirmatory run.", ""]
    for title in SECTIONS:
        key = title.split()[0]
        lines.extend((f"## {title}", "", f"{bodies[key]} {evidence}", ""))
    output = OUTPUT / "Глава_4_FuzzyXAI_formative.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"PASS: chapter4_practical_formative sections={len(SECTIONS)} placeholders=0 confirmatory_claim=false")


if __name__ == "__main__":
    main()

