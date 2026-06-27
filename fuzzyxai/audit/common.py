from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "reports" / "audit"


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
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def issue_dicts(issues: list[AuditIssue]) -> list[dict[str, Any]]:
    return [asdict(issue) for issue in issues]


def write_audit_reports(issues: list[AuditIssue], extra_sections: list[str] | None = None) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(AUDIT_DIR / "fuzzyxai_final_audit.json", {"commit": current_commit(), "issues": issue_dicts(issues)})
    counts = {level: sum(issue.severity == level for issue in issues) for level in ["BLOCKER", "MAJOR", "MINOR"]}
    lines = [
        "# FuzzyXAI final readiness audit",
        "",
        f"Commit: `{current_commit()}`",
        "",
        "## Краткий статус",
        "",
        f"- BLOCKER: {counts['BLOCKER']}",
        f"- MAJOR: {counts['MAJOR']}",
        f"- MINOR: {counts['MINOR']}",
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
