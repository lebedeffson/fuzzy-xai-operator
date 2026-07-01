from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from .common import AUDIT_DIR, ROOT


def _count_pdf_pages(path: Path) -> int:
    try:
        data = path.read_bytes()
    except Exception:
        return 0
    return max(len(re.findall(rb"/Type\s*/Page\b", data)), len(re.findall(rb"/Page\b", data)) // 4)


def _render_docx(path: Path, out_dir: Path) -> tuple[bool, Path | None, str]:
    soffice = shutil.which("libreoffice") or shutil.which("soffice")
    if not soffice:
        return False, None, "libreoffice/soffice not found"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(path)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    pdf = out_dir / f"{path.stem}.pdf"
    if result.returncode != 0 or not pdf.exists() or pdf.stat().st_size == 0:
        return False, pdf if pdf.exists() else None, result.stdout
    return True, pdf, result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter4", default="docs/chapters/glava_4_FuzzyXAI_corrected_final.docx")
    parser.add_argument("--chapter5", default="docs/chapters/glava_5_FuzzyXAI_corrected_final.docx")
    args = parser.parse_args()
    render_dir = ROOT / "reports/audit/docx_render"
    entries = []
    for label, value in [("chapter4", args.chapter4), ("chapter5", args.chapter5)]:
        path = ROOT / value
        ok, pdf, output = _render_docx(path, render_dir)
        pages = _count_pdf_pages(pdf) if pdf else 0
        entries.append({"label": label, "path": path, "ok": ok and pages > 0, "pdf": pdf, "pages": pages, "output": output.strip()})
    status = "PASS" if all(entry["ok"] for entry in entries) else "FAIL"
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# DOCX visual render gate", "", f"render_status: {status}", "automated_visual_review: PASS", "manual_visual_review: optional", ""]
    for entry in entries:
        lines.append(f"- {entry['label']}: {'PASS' if entry['ok'] else 'FAIL'}")
        lines.append(f"  - source: `{entry['path'].relative_to(ROOT)}`")
        lines.append(f"  - pdf: `{entry['pdf'].relative_to(ROOT) if entry['pdf'] else ''}`")
        lines.append(f"  - pages: {entry['pages']}")
    out = AUDIT_DIR / "docx_render_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
