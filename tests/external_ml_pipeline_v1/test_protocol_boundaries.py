from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_no_pipeline_specific_branch_in_core() -> None:
    diagnostics = ROOT / "framework/fuzzyxai/fuzzyxai/diagnostics"
    source = "\n".join(path.read_text(encoding="utf-8") for path in diagnostics.glob("*.py"))
    assert "ext1-" not in source and "ext2-" not in source and "ext3-" not in source and "ext4-" not in source


def test_no_docx_or_pdf_added_for_external_cycle() -> None:
    paths = list((ROOT / "reports/external_ml_pipeline_v1").glob("**/*")) if (ROOT / "reports/external_ml_pipeline_v1").exists() else []
    assert not [path for path in paths if path.suffix.lower() in {".docx", ".pdf"}]
