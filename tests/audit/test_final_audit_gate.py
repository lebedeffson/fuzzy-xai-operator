import json
import zipfile
from pathlib import Path


def test_final_audit_has_no_blockers_or_majors() -> None:
    report = json.loads(Path("reports/audit/fuzzyxai_final_audit.json").read_text(encoding="utf-8"))
    severities = [issue["severity"] for issue in report["issues"]]
    assert "BLOCKER" not in severities
    assert "MAJOR" not in severities
    assert report["source_commit"]
    assert report["artifact_commit"]
    assert "dirty" not in report["source_commit"]


def test_stale_terms_have_no_review_hits() -> None:
    report = Path("reports/audit/stale_terms_report.md").read_text(encoding="utf-8")
    assert "[review]" not in report


def test_docx_chapter_gate_passes() -> None:
    report = Path("reports/audit/docx_audit_report.md").read_text(encoding="utf-8")
    assert "status: PASS" in report


def test_audit_package_is_clean() -> None:
    forbidden = ["__pycache__", ".pyc", ".pytest_cache", ".venv", "node_modules", ".DS_Store"]
    with zipfile.ZipFile("fuzzyxai_final_audit_package.zip") as zf:
        names = zf.namelist()
    assert not [name for name in names if any(term in name for term in forbidden)]


def test_proof_package_release_metadata_is_explicit() -> None:
    proof = json.loads(Path("reports/studio_batch/hybrid_xiris_proof_package.json").read_text(encoding="utf-8"))
    assert proof["source_commit"]
    assert proof["artifact_commit"]
    assert proof["audit_branch"] == "audit/fuzzyxai-final-readiness"
    assert "dirty" not in proof["code_version"]
    assert isinstance(proof["working_tree_dirty_ignored_paths"], list)
    assert isinstance(proof["working_tree_dirty_unignored_paths"], list)
