from __future__ import annotations

import csv
import json
from dataclasses import asdict
from html import escape
from pathlib import Path

from .contracts import BatchDiagnosticReport, DiagnosticReport


def write_report_json(report: DiagnosticReport, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def write_report_html(report: DiagnosticReport, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    issues = "".join(
        f"<li><code>{escape(issue.code)}</code>: {escape(issue.symptom)} "
        f"<span>[{escape(', '.join(issue.evidence_refs) or 'нет ссылки')}]</span></li>"
        for issue in report.issues
    ) or "<li>Нарушения не обнаружены.</li>"
    steps = "".join(
        f"<li><strong>{escape(step.title)}</strong> ({escape(step.operation)}); "
        f"target={escape(step.target.key)}; approval={step.requires_human_approval}</li>"
        for step in (report.repair_plan.steps if report.repair_plan else ())
    ) or "<li>План не требуется или не сформирован.</li>"
    cut = ", ".join(report.minimal_cut.atom_keys) if report.minimal_cut else "не требуется"
    limitations = "".join(f"<li>{escape(item)}</li>" for item in report.limitations)
    output.write_text(
        "<!doctype html><html lang='ru'><meta charset='utf-8'>"
        "<title>FuzzyXAI diagnostic report</title>"
        "<style>body{font:16px Georgia,serif;max-width:1100px;margin:36px auto;color:#111}"
        "h1,h2{font-family:'Trebuchet MS',sans-serif}code{font:14px monospace}"
        "section{border-top:1px solid #777;padding:14px 0}.state{font-weight:bold}"
        ".marker{display:inline-block;border:2px solid #111;padding:2px 7px;margin-right:7px}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><body>"
        f"<h1>Диагностика маршрута {escape(report.route_id)}</h1>"
        f"<p class='state'><span class='marker'>СТАТУС</span>{escape(report.route_status)}</p>"
        f"<section><h2>Обнаруженные нарушения</h2><ul>{issues}</ul></section>"
        f"<section><h2>Диагностическая цепочка</h2><pre>{escape(report.expert_summary)}</pre></section>"
        f"<section><h2>Минимальный разрез</h2><p>{escape(cut)}</p></section>"
        f"<section><h2>План восстановления</h2><ol>{steps}</ol></section>"
        f"<section><h2>Повторная проверка</h2><p>{escape(report.recertification.status if report.recertification else 'не выполнялась')}</p></section>"
        f"<section><h2>Ограничения</h2><ul>{limitations}</ul></section>"
        f"<section><h2>Аудиторские сведения</h2><pre>{escape(report.audit_summary)}</pre>"
        f"<p>trace_sha256={escape(report.trace_sha256)}</p></section></body></html>\n",
        encoding="utf-8",
    )
    return output


def write_batch_csv(batch: BatchDiagnosticReport, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("route_id", "status", "issue_count", "cut_cost", "fully_executable", "trace_sha256"),
        )
        writer.writeheader()
        for report in batch.reports:
            writer.writerow(
                {
                    "route_id": report.route_id,
                    "status": report.route_status,
                    "issue_count": len(report.issues),
                    "cut_cost": report.minimal_cut.total_cost if report.minimal_cut else "",
                    "fully_executable": report.repair_plan.fully_executable if report.repair_plan else "",
                    "trace_sha256": report.trace_sha256,
                }
            )
    return output


def batch_to_dict(batch: BatchDiagnosticReport) -> dict[str, object]:
    payload = asdict(batch)
    payload["reports"] = [report.to_dict() for report in batch.reports]
    return payload
