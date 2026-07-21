#!/usr/bin/env python3
"""Build the Chapter 4 narrative shell; strong conclusions remain blocked."""

from __future__ import annotations

import json

from common import OUTPUT, prepare


SECTIONS = (
    "4.1 Научная задача и дизайн подтверждающего исследования",
    "4.2 Данные, модели, разбиения и воспроизводимость",
    "4.3 Корректная система аналогов",
    "4.4 Сохранение fidelity",
    "4.5 Устойчивость локальных объяснений и правил",
    "4.6 Сравнение с GAM, EBM, RuleFit и rule lists",
    "4.7 Неполнота происхождения и структурные разрывы",
    "4.8 Иерархия представлений и влияние на действие",
    "4.9 Selective observer и H3-v2",
    "4.10 Route validity и H5-A",
    "4.11 Planted и low-redundancy rules",
    "4.12 Чувствительность к компонентной сетке",
    "4.13 Робастность объяснения",
    "4.14 Масштабирование до миллионов объектов",
    "4.15 Formative AI-review и внешняя проверка",
    "4.16 Протокол воспроизведения",
    "4.17 Итоговые выводы и границы применимости",
)


def main() -> None:
    prepare()
    claims_path = OUTPUT / "chapter4_claims.json"
    if not claims_path.is_file():
        raise SystemExit("FAIL: build chapter4 claims before text")
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    if claims.get("final_chapter_allowed") is not False:
        raise SystemExit("FAIL: formative shell unexpectedly permits final claims")
    lines = [
        "# Глава 4. Формирующая вычислительная оболочка",
        "",
        "> Статус: FORMATIVE ONLY. Независимый confirmatory run не выполнен; новые положительные выводы запрещены.",
        "",
        "Исходные отрицательные результаты H3-original, H5-P-original и H6-general сохранены без изменений.",
        "",
    ]
    for section in SECTIONS:
        lines.extend([f"## {section}", "", "[PENDING CONFIRMATORY EVIDENCE]", ""])
    path = OUTPUT / "Глава_4_FuzzyXAI_formative_shell.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"PASS: chapter4_formative_shell sections={len(SECTIONS)} final_allowed=false")


if __name__ == "__main__":
    main()
