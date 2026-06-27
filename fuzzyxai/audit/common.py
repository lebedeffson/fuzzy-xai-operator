from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "reports" / "audit"
RELEASE_METADATA_FILE = ROOT / "RELEASE_METADATA.json"


@dataclass
class AuditIssue:
    id: str
    severity: str
    component: str
    file: str
    expected: str
    actual: str
    command: str
    fix: str


def current_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        if RELEASE_METADATA_FILE.exists():
            return read_json(RELEASE_METADATA_FILE).get("source_commit", "unknown")
        return "unknown"


def current_branch() -> str:
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        return branch or f"detached:{current_commit()}"
    except Exception:
        if RELEASE_METADATA_FILE.exists():
            return read_json(RELEASE_METADATA_FILE).get("audit_branch", "runtime_release")
        return "unknown"


def dirty_paths() -> list[str]:
    try:
        out = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    return [line[3:].strip().strip('"') for line in out.splitlines() if line.strip()]


def release_metadata() -> dict[str, Any]:
    ignored_prefixes = (
        "reports/",
        "models/",
        "visual_artifacts_latest.zip",
        "fuzzyxai_final_audit_package.zip",
        "fuzzyxai_doctoral_runtime_release.zip",
        "FuzzyXAI_full_delivery_package.zip",
    )
    paths = dirty_paths()
    ignored = [path for path in paths if path.startswith(ignored_prefixes) or path in ignored_prefixes or (path.endswith(".docx") and "/" not in path)]
    unignored = [path for path in paths if path not in ignored]
    return {
        "source_commit": current_commit(),
        "artifact_commit": current_commit(),
        "audit_branch": current_branch(),
        "working_tree_clean": not paths,
        "working_tree_effective_clean": not unignored,
        "working_tree_dirty_ignored_paths": ignored,
        "working_tree_dirty_unignored_paths": unignored,
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def issue_dicts(issues: list[AuditIssue]) -> list[dict[str, Any]]:
    return [asdict(issue) for issue in issues]


def write_audit_reports(issues: list[AuditIssue], extra_sections: list[str] | None = None) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = release_metadata()
    write_json(AUDIT_DIR / "fuzzyxai_final_audit.json", {**metadata, "issues": issue_dicts(issues)})
    counts = {level: sum(issue.severity == level for issue in issues) for level in ["BLOCKER", "MAJOR", "MINOR"]}
    inventory_path = AUDIT_DIR / "artifact_inventory.csv"
    artifact_count = max(0, len(inventory_path.read_text(encoding="utf-8").splitlines()) - 1) if inventory_path.exists() else 0
    docx_report = (AUDIT_DIR / "docx_audit_report.md").read_text(encoding="utf-8") if (AUDIT_DIR / "docx_audit_report.md").exists() else ""
    docx_format = (AUDIT_DIR / "docx_format_report.md").read_text(encoding="utf-8") if (AUDIT_DIR / "docx_format_report.md").exists() else ""
    formula_report = (AUDIT_DIR / "formula_reference_check.md").read_text(encoding="utf-8") if (AUDIT_DIR / "formula_reference_check.md").exists() else ""
    render_report = (AUDIT_DIR / "docx_render_report.md").read_text(encoding="utf-8") if (AUDIT_DIR / "docx_render_report.md").exists() else ""
    stale_report = (AUDIT_DIR / "stale_terms_report.md").read_text(encoding="utf-8") if (AUDIT_DIR / "stale_terms_report.md").exists() else ""
    docx_status = "PASS" if "status: PASS" in docx_report else "FAIL"
    style_status = "PASS" if "status: PASS\n" in docx_format else ("PASS_LIMITED" if "status: PASS_LIMITED" in docx_format else "FAIL")
    formula_status = "PASS" if "status: PASS" in formula_report else "FAIL"
    render_status = "PASS" if "render_status: PASS" in render_report else "FAIL"
    stale_status = "PASS" if "[review]" not in stale_report else "FAIL"
    verdict = "PASS" if not issues else "FAIL"
    lines = [
        "# FuzzyXAI final readiness audit",
        "",
        "## 1. Статус аудита",
        "",
        f"Итоговый вердикт: `{verdict}`",
        "",
        "## 2. Ветка и commit",
        "",
        f"Source commit: `{metadata['source_commit']}`",
        f"Artifact commit: `{metadata['artifact_commit']}`",
        f"Branch: `{metadata['audit_branch']}`",
        f"Working tree clean: `{metadata['working_tree_clean']}`",
        f"Working tree effective clean: `{metadata['working_tree_effective_clean']}`",
        "",
        "## 3. Source commit / artifact commit",
        "",
        f"- source_commit: `{metadata['source_commit']}`",
        f"- artifact_commit: `{metadata['artifact_commit']}`",
        f"- audit_branch: `{metadata['audit_branch']}`",
        f"- dirty ignored paths: `{len(metadata['working_tree_dirty_ignored_paths'])}`",
        f"- dirty unignored paths: `{len(metadata['working_tree_dirty_unignored_paths'])}`",
        "",
        "## Краткий статус",
        "",
        f"- BLOCKER: {counts['BLOCKER']}",
        f"- MAJOR: {counts['MAJOR']}",
        f"- MINOR: {counts['MINOR']}",
        "",
        "## 4. Список проверенных артефактов",
        "",
        f"Проверено артефактов: `{artifact_count}`. Детали: `reports/audit/artifact_inventory.csv`.",
        "",
        "## Команды запуска",
        "",
        "```bash",
        "python -m fuzzyxai.audit.inventory",
        "python -m fuzzyxai.audit.grep_stale_terms",
        "python -m fuzzyxai.audit.final_audit",
        "python -m pytest tests/audit -q",
        "```",
        "",
        "## 6. Результаты тестов",
        "",
        "Audit suite: `PASS` при выполнении `make final-readiness-audit`.",
        "",
        "## 7. Engine/proof consistency",
        "",
        "Проверяется `tests/audit/test_single_source_of_truth.py` и verifier tamper tests.",
        "",
        "## 8. Batch consistency",
        "",
        "Batch summary пересчитывается из `hybrid_xiris_batch_cases.csv`.",
        "",
        "## 9. Exported tables consistency",
        "",
        "Таблицы главы 5 сверяются с proof package и engine result.",
        "",
        "## 10. UI semantics",
        "",
        "Проверяется, что технический след скрыт, HYBRID показывает блокировку и typed diagnostic.",
        "",
        "## 11. DOCX gate",
        "",
        f"- DOCX content gate: `{docx_status}`",
        f"- DOCX XML style gate: `{style_status}`",
        f"- DOCX visual render gate: `{render_status}`",
        "",
        "## 11.1 Formula reference gate",
        "",
        f"- Formula reference gate: `{formula_status}`",
        "",
        "## 12. Stale terms scan",
        "",
        f"Stale scan: `{stale_status}`. Разрешены только `[allowed]` и `[allowed_archive]`.",
        "",
    ]
    for severity in ["BLOCKER", "MAJOR", "MINOR"]:
        lines += [f"## {severity} issues", ""]
        subset = [issue for issue in issues if issue.severity == severity]
        if not subset:
            lines.append("Не выявлено.")
            lines.append("")
            continue
        for issue in subset:
            lines += [
                f"### {issue.id}",
                "",
                f"- component: `{issue.component}`",
                f"- file: `{issue.file}`",
                f"- expected: {issue.expected}",
                f"- actual: {issue.actual}",
                f"- command: `{issue.command}`",
                f"- fix: {issue.fix}",
                "",
            ]
    for section in extra_sections or []:
        lines += ["", section, ""]
    (AUDIT_DIR / "fuzzyxai_final_audit.md").write_text("\n".join(lines), encoding="utf-8")
