from __future__ import annotations

import re
from pathlib import Path

from .common import AUDIT_DIR, ROOT


TERMS: list[tuple[str, re.Pattern[str]]] = [
    ("rho:0.74", re.compile(r'("rho"\s*:\s*0\.74\b|rho\s*=\s*0\.74\b)')),
    ("delta:0.08", re.compile(r'"delta"\s*:\s*0\.08\b')),
    ("gamma_ij:0.35", re.compile(r'"gamma_ij"\s*:\s*0\.35\b')),
    ("D_source_conflict", re.compile(r"D_source_conflict")),
    ("gamma_max:0.45", re.compile(r'"gamma_max"\s*:\s*0\.45\b')),
    ("delta_max:0.15", re.compile(r'"delta_max"\s*:\s*0\.15\b')),
    ("v0.9", re.compile(r"v0\.9")),
    ("JSON как пользовательский интерфейс", re.compile(r"JSON как пользовательский интерфейс")),
]


ARCHIVE_PREFIXES = (
    "reports/reproducibility_artifacts/",
    "reports/unified_full_demo/",
    "reports/chapter5/real_data_validation/",
)


def _context(lines: list[str], index: int) -> str:
    start = max(0, index - 45)
    return "\n".join(lines[start : index + 1])


def _classification(term: str, rel: Path, line: str, context: str) -> str:
    rel_s = str(rel)
    if term == "D_source_conflict":
        return "allowed" if ("legacy_id" in line or "legacy_diagnostic_id" in line or "legacy_diagnostic_id=" in line) else "review"
    if rel_s.startswith(ARCHIVE_PREFIXES) and term in {"gamma_max:0.45", "delta_max:0.15"}:
        return "allowed_archive"
    if term == "delta:0.08" and ("gis_integro" in rel_s or '"gis_integro"' in context or "gamma_route" in context):
        return "allowed"
    return "review"


def scan() -> list[dict[str, str]]:
    roots = ["fuzzyxai", "apps", "configs", "reports", "patches", "tests"]
    hits: list[dict[str, str]] = []
    for root_name in roots:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix in {".pyc", ".png", ".pdf", ".zip", ".docx"}:
                continue
            rel = path.relative_to(ROOT)
            if str(rel).startswith(("fuzzyxai/audit/", "tests/audit/", "reports/audit/")):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            lines = text.splitlines()
            for term, pattern in TERMS:
                for line_no, line in enumerate(lines, start=1):
                    if pattern.search(line):
                        classification = _classification(term, rel, line, _context(lines, line_no - 1))
                        hits.append(
                            {
                                "term": term,
                                "path": str(path.relative_to(ROOT)),
                                "line": str(line_no),
                                "status": classification,
                                "allowed": str(classification in {"allowed", "allowed_archive"}),
                                "context": line.strip()[:220],
                            }
                        )
    return hits


def main() -> None:
    hits = scan()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Stale terms report", ""]
    if not hits:
        lines.append("Подозрительных старых терминов не найдено.")
    else:
        for hit in hits:
            flag = hit.get("status", "allowed" if hit["allowed"] == "True" else "review")
            lines.append(f"- `{hit['term']}` in `{hit['path']}:{hit['line']}` [{flag}]: {hit['context']}")
    path = AUDIT_DIR / "stale_terms_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
