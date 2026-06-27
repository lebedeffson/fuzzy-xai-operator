from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

from .common import AUDIT_DIR


def _read_xml(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.read("word/document.xml").decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _style_counts(path: Path) -> dict[str, int]:
    xml = _read_xml(path)
    return {
        "Heading1": xml.count('w:val="Heading1"'),
        "Heading2": xml.count('w:val="Heading2"'),
        "Heading3": xml.count('w:val="Heading3"'),
        "Caption": xml.count('w:val="Caption"'),
        "tables": xml.count("<w:tbl>"),
        "drawings": xml.count("<w:drawing"),
        "empty_pages_hint": xml.count('w:type="page"') if "<w:lastRenderedPageBreak" in xml else 0,
    }


def _text(path: Path) -> str:
    xml = _read_xml(path)
    return re.sub(r"<[^>]+>", " ", xml)


def _docx_style_status(path: Path) -> tuple[str, list[str]]:
    counts = _style_counts(path)
    text = _text(path)
    issues: list[str] = []
    if counts["Heading1"] < 1:
        issues.append("missing Heading1")
    if counts["Heading2"] < 1:
        issues.append("missing Heading2")
    if re.search(r"\b[45]\.\d+\.\d+", text) and counts["Heading3"] < 1:
        issues.append("document has x.y.z headings but no Heading3 style")
    if ("Рисунок" in text or "Таблица" in text) and counts["Caption"] < 1:
        issues.append("captions text exists but no Caption style")
    if counts["drawings"] and counts["Caption"] < counts["drawings"]:
        issues.append("fewer Caption paragraphs than drawings")
    return ("PASS" if not issues else "FAIL", issues)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter4", default="docs/chapters/glava_4_FuzzyXAI_corrected_final.docx")
    parser.add_argument("--chapter5", default="docs/chapters/glava_5_FuzzyXAI_corrected_final.docx")
    args = parser.parse_args()
    sources = []
    for value in [args.chapter4, args.chapter5]:
        path = Path(value)
        fallback = path.with_suffix(path.suffix + ".audit.txt")
        if path.exists():
            sources.append((path, "docx"))
        elif fallback.exists():
            sources.append((fallback, "audit_txt"))
        else:
            sources.append((path, "missing"))
    docx_results = []
    for source, kind in sources:
        if kind == "docx":
            docx_results.append((source, *_docx_style_status(source)))
    if all(kind == "docx" for _, kind in sources) and all(status == "PASS" for _, status, _ in docx_results):
        status = "PASS"
    elif all(kind in {"docx", "audit_txt"} for _, kind in sources):
        status = "PASS_LIMITED"
    else:
        status = "FAIL"
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIT_DIR / "docx_format_report.md"
    lines = [
        "# DOCX format audit",
        "",
        f"status: {status}",
        "",
        "DOCX XML gate checks Heading styles, Caption styles, table presence and drawing/caption consistency. Full visual page-break verification still requires manual Word/LibreOffice inspection.",
    ]
    for source, kind in sources:
        lines.append(f"- {kind}: `{source}`")
        if kind == "docx":
            s, issues = _docx_style_status(source)
            counts = _style_counts(source)
            lines.append(f"  - style_status: {s}")
            lines.append(f"  - counts: {counts}")
            if issues:
                lines.append(f"  - issues: {issues}")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
