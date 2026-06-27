from __future__ import annotations

import argparse
from pathlib import Path

from .common import AUDIT_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter4")
    parser.add_argument("--chapter5")
    args = parser.parse_args()
    lines = ["# DOCX chapter audit", ""]
    for label, value in [("chapter4", args.chapter4), ("chapter5", args.chapter5)]:
        path = Path(value) if value else None
        if not path or not path.exists():
            lines.append(f"- {label}: missing; audit skipped.")
        else:
            lines.append(f"- {label}: found `{path}`; detailed DOCX semantic audit requires final text lock.")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out = AUDIT_DIR / "docx_audit_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
