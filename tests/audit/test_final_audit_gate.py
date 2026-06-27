import json
from pathlib import Path


def test_final_audit_has_no_blockers_or_majors() -> None:
    report = json.loads(Path("reports/audit/fuzzyxai_final_audit.json").read_text(encoding="utf-8"))
    severities = [issue["severity"] for issue in report["issues"]]
    assert "BLOCKER" not in severities
    assert "MAJOR" not in severities


def test_stale_terms_have_no_review_hits() -> None:
    report = Path("reports/audit/stale_terms_report.md").read_text(encoding="utf-8")
    assert "[review]" not in report


def test_docx_chapter_gate_passes() -> None:
    report = Path("reports/audit/docx_audit_report.md").read_text(encoding="utf-8")
    assert "status: PASS" in report
