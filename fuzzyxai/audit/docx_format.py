from __future__ import annotations

from .common import AUDIT_DIR


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIT_DIR / "docx_format_report.md"
    path.write_text("# DOCX format audit\n\nFinal chapter DOCX files are required for layout/style verification.\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
