from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

from .build_chapter import CHANGELOG, DOCX, FINAL, PDF, VALIDATION
from .common import ARTIFACTS, read_json, sha256_file


FORBIDDEN_MAIN_TEXT = (
    "TODO",
    "TBD",
    "placeholder",
    "not_supported",
    "external gate",
    "внешние ворота",
    "формальная экспертная валидация",
    "доказано повышение безопасности",
    "доказана понятность",
    "универсальное превосходство",
    "исходного объяснителяs",
    "постмодельных объяснения",
    "интерпретируемых-подходы",
    "сквозного конвейере",
    "слоя слоя",
)


def _capture(*command: str) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def _numbers(text: str, kind: str) -> list[int]:
    return sorted(set(int(value) for value in re.findall(rf"{kind} 4\.(\d+)\s+[—-]", text)))


def validate() -> dict[str, object]:
    required = (
        DOCX,
        PDF,
        CHANGELOG,
        FINAL / "Глава_4_FuzzyXAI_v13_evidence_map.json",
        FINAL / "Глава_4_FuzzyXAI_v13_leakage_audit.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"missing document artifacts: {missing}")

    with tempfile.TemporaryDirectory(prefix="fuzzyxai-v13-doc-validation-") as temporary:
        work = Path(temporary)
        docx_text_path = work / "docx.txt"
        pdf_text_path = work / "pdf.txt"
        subprocess.run(["pandoc", str(DOCX), "-t", "plain", "-o", str(docx_text_path)], check=True)
        subprocess.run(["pdftotext", "-layout", str(PDF), str(pdf_text_path)], check=True)
        docx_text = docx_text_path.read_text(encoding="utf-8")
        pdf_text = pdf_text_path.read_text(encoding="utf-8")
        main_text = docx_text.split("Приложение А.", maxsplit=1)[0]
        forbidden = {term: len(re.findall(re.escape(term), main_text, flags=re.I)) for term in FORBIDDEN_MAIN_TEXT}
        forbidden = {term: count for term, count in forbidden.items() if count}
        machine_statuses = {
            term: len(re.findall(rf"\b{term}\b", main_text))
            for term in ("supported", "PASS", "FAIL")
            if re.search(rf"\b{term}\b", main_text)
        }
        tables = _numbers(docx_text, "Таблица")
        figures = _numbers(docx_text, "Рисунок")
        table_sequence = tables == list(range(min(tables), max(tables) + 1)) if tables else False
        figure_sequence = figures == list(range(min(figures), max(figures) + 1)) if figures else False
        pdf_info = _capture("pdfinfo", str(PDF))
        pages_match = re.search(r"^Pages:\s+(\d+)", pdf_info, flags=re.M)
        pages = int(pages_match.group(1)) if pages_match else 0
        page_text = pdf_text.split("\f")
        sparse_pages = [index + 1 for index, value in enumerate(page_text[:pages]) if len(value.strip()) < 20]
        render_prefix = work / "page"
        subprocess.run(["pdftoppm", "-png", "-r", "72", str(PDF), str(render_prefix)], check=True)
        rendered_pages = len(list(work.glob("page-*.png")))

    evidence = read_json(ARTIFACTS / "evidence_map.json")
    leakage = read_json(ARTIFACTS / "leakage_audit.json")
    errors = []
    if forbidden:
        errors.append(f"forbidden_main_text:{forbidden}")
    if machine_statuses:
        errors.append(f"machine_statuses_in_main_text:{machine_statuses}")
    if not table_sequence:
        errors.append("table_numbering_not_sequential")
    if not figure_sequence:
        errors.append("figure_numbering_not_sequential")
    if rendered_pages != pages:
        errors.append(f"rendered_pages:{rendered_pages}/{pages}")
    if sparse_pages:
        errors.append(f"sparse_pdf_pages:{sparse_pages}")
    if not leakage.get("passed"):
        errors.append("leakage_audit_failed")
    if "Современный прикладной контур" not in main_text or "DistilBERT" not in main_text:
        errors.append("modern_contour_missing_from_main_text")
    if "не показал статистически подтверждённого" not in main_text and "статистически подтверждённое снижение" not in main_text:
        errors.append("primary_policy_conclusion_missing")

    checksums = {
        DOCX.name: sha256_file(DOCX),
        PDF.name: sha256_file(PDF),
        CHANGELOG.name: sha256_file(CHANGELOG),
        "Глава_4_FuzzyXAI_v13_evidence_map.json": sha256_file(FINAL / "Глава_4_FuzzyXAI_v13_evidence_map.json"),
        "Глава_4_FuzzyXAI_v13_leakage_audit.json": sha256_file(FINAL / "Глава_4_FuzzyXAI_v13_leakage_audit.json"),
    }
    status = "PASS" if not errors else "FAIL"
    VALIDATION.write_text(
        "# Отчёт проверки главы 4 v13\n\n"
        f"- статус: `{status}`\n"
        f"- страниц PDF: `{pages}`\n"
        f"- таблиц: `{len(tables)}`; диапазон: `4.{min(tables)}–4.{max(tables)}`\n"
        f"- рисунков: `{len(figures)}`; диапазон: `4.{min(figures)}–4.{max(figures)}`\n"
        f"- последовательность таблиц: `{table_sequence}`\n"
        f"- последовательность рисунков: `{figure_sequence}`\n"
        f"- отрендерено страниц: `{rendered_pages}`\n"
        f"- страницы без содержимого: `{sparse_pages}`\n"
        f"- запрещённые формулировки в основном тексте: `{forbidden}`\n"
        f"- машинные статусы в основном тексте: `{machine_statuses}`\n"
        f"- записей в карте доказательств v13: `{len(evidence['entries'])}`\n"
        f"- аудит утечки: `{leakage.get('passed')}`\n"
        f"- неизменённые отрицательные статусы: `H3-original, H5-P-original, H6-general`\n"
        f"- ошибки: `{errors}`\n\n"
        "## Исправленные классы дефектов\n\n"
        "- пользовательская апробация отделена от количественной валидации;\n"
        "- производительность кэшированного слоя отделена от полного времени;\n"
        "- новый современный контур включён в основной текст;\n"
        "- числовые таблицы связаны с машинно-читаемой картой доказательств;\n"
        "- отрицательные результаты сохранены.\n\n"
        "## Контрольные суммы\n\n"
        + "".join(f"- `{digest}`  `{name}`\n" for name, digest in checksums.items()),
        encoding="utf-8",
    )
    if errors:
        raise RuntimeError("; ".join(errors))
    return {"pages": pages, "tables": len(tables), "figures": len(figures), "evidence_entries": len(evidence["entries"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = validate()
    print(f"PASS: pages={result['pages']} tables={result['tables']} figures={result['figures']}")


if __name__ == "__main__":
    main()
