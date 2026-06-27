from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from fuzzyxai.core.proof_package import verify_proof_package

from .common import AUDIT_DIR, ROOT, AuditIssue, read_json, write_audit_reports
from .grep_stale_terms import scan as scan_stale
from .inventory import build_inventory


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def collect_issues() -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    proof_path = ROOT / "reports/studio_batch/hybrid_xiris_proof_package.json"
    if proof_path.exists():
        package = read_json(proof_path)
        verification = verify_proof_package(package)
        if not verification.valid:
            issues.append(
                AuditIssue(
                    "FXAI-AUDIT-001",
                    "BLOCKER",
                    "proof_package",
                    str(proof_path.relative_to(ROOT)),
                    "proof package verifier returns valid=true",
                    "; ".join(verification.errors),
                    "python -m fuzzyxai.audit.final_audit",
                    "Rebuild proof package from engine and repair verifier errors.",
                )
            )
    else:
        issues.append(AuditIssue("FXAI-AUDIT-001", "BLOCKER", "proof_package", str(proof_path), "file exists", "missing", "python -m fuzzyxai.run_scenario hybrid_xiris --batch", "Regenerate batch proof package."))

    cases_path = ROOT / "reports/studio_batch/hybrid_xiris_batch_cases.csv"
    summary_path = ROOT / "reports/studio_batch/hybrid_xiris_batch_summary.json"
    if cases_path.exists() and summary_path.exists():
        rows = _csv_rows(cases_path)
        summary = read_json(summary_path)
        counts = {
            "accept": sum(row.get("fuzzyxai_action") == "accept" for row in rows),
            "lower_confidence": sum(row.get("fuzzyxai_action") == "lower_confidence" for row in rows),
            "block": sum(row.get("fuzzyxai_action") == "block" for row in rows),
            "baseline_critical_misses": sum(int(row.get("baseline_miss", 0)) for row in rows),
            "fuzzyxai_critical_misses": sum(int(row.get("fuzzyxai_miss", 0)) for row in rows),
        }
        for key, expected in counts.items():
            if int(summary.get(key, -1)) != expected:
                issues.append(AuditIssue("FXAI-AUDIT-002", "BLOCKER", "batch", str(summary_path.relative_to(ROOT)), f"{key} == count from cases", f"{summary.get(key)} != {expected}", "python -m fuzzyxai.audit.final_audit", "Compute summary only from cases CSV."))
    else:
        issues.append(AuditIssue("FXAI-AUDIT-002", "BLOCKER", "batch", "reports/studio_batch", "cases and summary exist", "missing", "python -m fuzzyxai.run_scenario hybrid_xiris --batch", "Regenerate batch outputs."))

    d_e = ROOT / "reports/chapter5/studio_tables/table_5_4_dE.csv"
    if d_e.exists():
        rows = _csv_rows(d_e)
        required = {"component", "value", "weight", "contribution", "definition"}
        actual = set(rows[0]) if rows else set()
        if not required <= actual:
            issues.append(AuditIssue("FXAI-AUDIT-003", "MAJOR", "chapter5_tables", str(d_e.relative_to(ROOT)), f"columns include {sorted(required)}", f"columns are {sorted(actual)}", "python -m fuzzyxai.export_tables --scenario hybrid_xiris", "Add contribution column and regenerate table."))
    else:
        issues.append(AuditIssue("FXAI-AUDIT-003", "MAJOR", "chapter5_tables", str(d_e.relative_to(ROOT)), "table exists", "missing", "python -m fuzzyxai.export_tables --scenario hybrid_xiris", "Regenerate chapter 5 tables."))

    for row in build_inventory():
        if row["status"] != "ok":
            issues.append(AuditIssue("FXAI-AUDIT-004", "MAJOR", "inventory", row["artifact_path"], "artifact exists, non-empty, schema ok", str(row), "python -m fuzzyxai.audit.inventory", "Restore missing artifact or repair schema."))

    bad_stale = [hit for hit in scan_stale() if hit.get("status") == "review"]
    if bad_stale:
        issues.append(AuditIssue("FXAI-AUDIT-005", "MAJOR", "stale_terms", "repository", "no unapproved stale terms", f"{len(bad_stale)} hits", "python -m fuzzyxai.audit.grep_stale_terms", "Review stale terms report and either update text or mark legacy explicitly."))

    chapter4 = ROOT / "docs/chapters/glava_4_FuzzyXAI_corrected_final.docx"
    chapter5 = ROOT / "docs/chapters/glava_5_FuzzyXAI_corrected_final.docx"
    chapter4_ok = chapter4.exists() or chapter4.with_suffix(chapter4.suffix + ".audit.txt").exists()
    chapter5_ok = chapter5.exists() or chapter5.with_suffix(chapter5.suffix + ".audit.txt").exists()
    if not chapter4_ok or not chapter5_ok:
        issues.append(AuditIssue("FXAI-AUDIT-006", "MAJOR", "docx", "chapter4/chapter5 DOCX", "chapter DOCX files or .docx.audit.txt exports present for formula/style audit", f"chapter4={chapter4_ok}, chapter5={chapter5_ok}", "python -m fuzzyxai.audit.docx_chapters --chapter4 docs/chapters/glava_4_FuzzyXAI_corrected_final.docx --chapter5 docs/chapters/glava_5_FuzzyXAI_corrected_final.docx", "Provide final chapter DOCX files or audited text exports and rerun DOCX audit."))
    docx_report = AUDIT_DIR / "docx_audit_report.md"
    if docx_report.exists() and "status: PASS" not in docx_report.read_text(encoding="utf-8"):
        issues.append(AuditIssue("FXAI-AUDIT-007", "MAJOR", "docx", str(docx_report.relative_to(ROOT)), "DOCX content gate PASS", "FAIL", "python -m fuzzyxai.audit.docx_chapters --chapter4 docs/chapters/glava_4_FuzzyXAI_corrected_final.docx --chapter5 docs/chapters/glava_5_FuzzyXAI_corrected_final.docx", "Align chapter text with verified engine values."))
    docx_format = AUDIT_DIR / "docx_format_report.md"
    if docx_format.exists() and "status: PASS\n" not in docx_format.read_text(encoding="utf-8"):
        issues.append(AuditIssue("FXAI-AUDIT-008", "MAJOR", "docx_format", str(docx_format.relative_to(ROOT)), "Real DOCX style gate PASS", "FAIL or PASS_LIMITED", "python -m fuzzyxai.audit.docx_format --chapter4 docs/chapters/glava_4_FuzzyXAI_corrected_final.docx --chapter5 docs/chapters/glava_5_FuzzyXAI_corrected_final.docx", "Use real DOCX files with Heading/Caption styles."))

    return issues


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "fuzzyxai.audit.inventory"], cwd=ROOT, check=False)
    subprocess.run([sys.executable, "-m", "fuzzyxai.audit.grep_stale_terms"], cwd=ROOT, check=False)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "fuzzyxai.audit.docx_chapters",
            "--chapter4",
            "docs/chapters/glava_4_FuzzyXAI_corrected_final.docx",
            "--chapter5",
            "docs/chapters/glava_5_FuzzyXAI_corrected_final.docx",
        ],
        cwd=ROOT,
        check=False,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "fuzzyxai.audit.docx_format",
            "--chapter4",
            "docs/chapters/glava_4_FuzzyXAI_corrected_final.docx",
            "--chapter5",
            "docs/chapters/glava_5_FuzzyXAI_corrected_final.docx",
        ],
        cwd=ROOT,
        check=False,
    )
    issues = collect_issues()
    write_audit_reports(issues)
    print(AUDIT_DIR / "fuzzyxai_final_audit.md")


if __name__ == "__main__":
    main()
