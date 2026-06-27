from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

from .common import AUDIT_DIR


REQUIRED_CHAPTER5 = [
    "γ = 0,351",
    "Δ = 0,106811",
    "r_Δ = 0,3225",
    "ρ = 0,800",
    "D_quality_source_conflict",
    "612",
    "201",
    "187",
    "168",
    "max(|p - α_mean|, |p - s|)",
]

FORBIDDEN = [
    "ρ = 0,74",
    "rho = 0.74",
    "Δ = 0,08",
    "γ_ij = 0,35",
    "v0.9",
    "JSON как пользовательский интерфейс",
]


def _fallback(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".audit.txt")


def _source(path_arg: str | None, default: str) -> tuple[str, Path | None]:
    path = Path(path_arg or default)
    if path.exists():
        return "docx", path
    fallback = _fallback(path)
    if fallback.exists():
        return "audit_txt", fallback
    return "missing", None


def _read_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
        return re.sub(r"<[^>]+>", " ", xml)
    except Exception:
        return ""


def _read_source(kind: str, path: Path | None) -> str:
    if not path:
        return ""
    if kind == "docx":
        return _read_docx(path)
    return path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter4")
    parser.add_argument("--chapter5")
    args = parser.parse_args()
    ch4_kind, ch4_path = _source(args.chapter4, "docs/chapters/glava_4_FuzzyXAI_corrected_final.docx")
    ch5_kind, ch5_path = _source(args.chapter5, "docs/chapters/glava_5_FuzzyXAI_corrected_final.docx")
    ch5_text = _read_source(ch5_kind, ch5_path)
    missing_required = [term for term in REQUIRED_CHAPTER5 if term not in ch5_text]
    forbidden_found = [term for term in FORBIDDEN if term in ch5_text]
    status = "PASS" if ch4_path and ch5_path and not missing_required and not forbidden_found else "FAIL"
    lines = [
        "# DOCX chapter audit",
        "",
        f"status: {status}",
        f"chapter4_source: {ch4_kind} `{ch4_path}`",
        f"chapter5_source: {ch5_kind} `{ch5_path}`",
        "",
        "## Required chapter 5 terms",
    ]
    for term in REQUIRED_CHAPTER5:
        lines.append(f"- {term}: {'OK' if term in ch5_text else 'MISSING'}")
    lines.append("")
    lines.append("## Forbidden terms")
    for term in FORBIDDEN:
        lines.append(f"- {term}: {'FOUND' if term in ch5_text else 'OK'}")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out = AUDIT_DIR / "docx_audit_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
