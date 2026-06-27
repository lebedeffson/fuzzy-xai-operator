from __future__ import annotations

import argparse
from pathlib import Path

from .common import AUDIT_DIR


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
    status = "PASS_LIMITED" if all(kind in {"docx", "audit_txt"} for _, kind in sources) else "FAIL"
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIT_DIR / "docx_format_report.md"
    lines = [
        "# DOCX format audit",
        "",
        f"status: {status}",
        "",
        "Audit text exports validate terms/numbers but cannot prove Word styles. Replace with DOCX for Heading/Caption checks before final submission.",
    ]
    for source, kind in sources:
        lines.append(f"- {kind}: `{source}`")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
