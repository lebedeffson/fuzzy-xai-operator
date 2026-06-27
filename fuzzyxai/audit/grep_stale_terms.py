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
            for term, pattern in TERMS:
                for line_no, line in enumerate(text.splitlines(), start=1):
                    if pattern.search(line):
                        allowed = term == "D_source_conflict" and ("legacy" in line or "legacy_id" in text)
                        hits.append(
                            {
                                "term": term,
                                "path": str(path.relative_to(ROOT)),
                                "line": str(line_no),
                                "allowed": str(allowed),
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
            flag = "allowed" if hit["allowed"] == "True" else "review"
            lines.append(f"- `{hit['term']}` in `{hit['path']}:{hit['line']}` [{flag}]: {hit['context']}")
    path = AUDIT_DIR / "stale_terms_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
