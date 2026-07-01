from __future__ import annotations

import json
from pathlib import Path

from .common import AUDIT_DIR, ROOT
from .docx_chapters import _read_source, _source


REQUIRED = ["2.7", "2.14", "2.21", "2.30", "3.29", "3.42", "3.44"]
INDEX = ROOT / "docs/chapters/formula_index_ch2_ch3.json"


def check_formula_references() -> tuple[bool, list[str]]:
    messages: list[str] = []
    if not INDEX.exists():
        return False, ["formula index missing"]
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    ok = True
    for number in REQUIRED:
        item = index.get(number)
        if not item or item.get("status") != "ok":
            ok = False
            messages.append(f"({number}) — FAIL")
        else:
            messages.append(f"({number}) — OK — {item.get('title', '')}")
    ch4_kind, ch4_path = _source("docs/chapters/glava_4_FuzzyXAI_corrected_final.docx", "")
    ch5_kind, ch5_path = _source("docs/chapters/glava_5_FuzzyXAI_corrected_final.docx", "")
    text = _read_source(ch4_kind, ch4_path) + "\n" + _read_source(ch5_kind, ch5_path)
    for forbidden in ["(2.999)", "(3.999)"]:
        if forbidden in text:
            ok = False
            messages.append(f"{forbidden} — FAIL unexpected reference")
    return ok, messages


def main() -> None:
    ok, messages = check_formula_references()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Formula reference check",
        "",
        f"status: {'PASS' if ok else 'FAIL'}",
        "",
        "Formula index: `docs/chapters/formula_index_ch2_ch3.json`",
        "",
    ]
    lines.extend(f"- {message}" for message in messages)
    out = AUDIT_DIR / "formula_reference_check.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
