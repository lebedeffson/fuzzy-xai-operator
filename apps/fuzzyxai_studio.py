"""FuzzyXAI Studio: clickable operator-route GUI for chapters 2-5."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fuzzyxai.studio.operator_scenarios import STATUS_COLOR, build_report, ensure_scenario_json_files, load_scenarios

REPORT_DIR = ROOT / "reports" / "studio"


def _status_color(status: str) -> str:
    return STATUS_COLOR.get(str(status), "#334155")


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value))


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _rows(mapping: dict[str, Any]) -> list[dict[str, str]]:
    return [{"key": str(k), "value": _fmt(v)} for k, v in (mapping or {}).items()]


def _scenario_summary_rows(scenario: dict[str, Any]) -> list[dict[str, str]]:
    return _rows(scenario.get("summary", {}))


def _report_summary(report: dict[str, Any]) -> str:
    diagnostics = report.get("diagnostics", [])
    diag_text = ", ".join(str(d.get("type", "")) for d in diagnostics) or "нет"
    return (
        f"scenario={report.get('scenario_id')}; "
        f"action={report.get('final_action')}; "
        f"reason={report.get('action_reason')}; "
        f"diagnostics={diag_text}"
    )


def _action_chart_option(summary: dict[str, Any]) -> dict[str, Any]:
    labels = [k for k in ["accept", "lower_confidence", "request_more_data", "defer_to_human", "block"] if k in summary]
    return {
        "tooltip": {"trigger": "item"},
        "grid": {"left": 44, "right": 20, "top": 18, "bottom": 52},
        "xAxis": {"type": "category", "data": labels, "axisLabel": {"interval": 0, "rotate": 14}},
        "yAxis": {"type": "value"},
        "series": [
            {
                "type": "bar",
                "data": [summary.get(k, 0) for k in labels],
                "itemStyle": {"color": "#0f766e"},
                "label": {"show": True, "position": "top"},
            }
        ],
    }


def _miss_chart_option(summary: dict[str, Any]) -> dict[str, Any]:
    if "baseline_critical_misses" in summary:
        labels = ["baseline", "FuzzyXAI"]
        values = [summary.get("baseline_critical_misses", 0), summary.get("fuzzyxai_critical_misses", 0)]
    elif "checks_without_beacon" in summary:
        labels = ["without BEACON", "with BEACON", "audit reports"]
        values = [summary.get("checks_without_beacon", 0), summary.get("checks_with_beacon", 0), summary.get("audit_reports", 0)]
    else:
        labels = [k for k, v in summary.items() if isinstance(v, (int, float))]
        values = [summary.get(k, 0) for k in labels]
    return {
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 52, "right": 20, "top": 18, "bottom": 58},
        "xAxis": {"type": "category", "data": labels, "axisLabel": {"interval": 0, "rotate": 18}},
        "yAxis": {"type": "value"},
        "series": [{"type": "bar", "data": values, "itemStyle": {"color": "#b45309"}, "label": {"show": True, "position": "top"}}],
    }


def _pipeline_status_counts(scenario: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in scenario.get("pipeline", []):
        status = str(node.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def run(port: int = 8097) -> None:
    try:
        from nicegui import app, ui
    except Exception as exc:  # pragma: no cover
        raise SystemExit("FuzzyXAI Studio requires nicegui. Install requirements.txt first.") from exc

    ensure_scenario_json_files()
    scenarios = load_scenarios()
    scenario_by_id = {s["scenario_id"]: s for s in scenarios}
    current = {"scenario_id": scenarios[0]["scenario_id"], "node_id": scenarios[0]["pipeline"][0]["node_id"]}

    app.add_static_files("/studio_reports", str(REPORT_DIR))
    ui.add_head_html(
        """
        <script>
          window.MathJax = {tex: {inlineMath: [['\\\\(','\\\\)'], ['$', '$']]}};
        </script>
        <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <style>
          :root { --fx-border:#d7dde8; --fx-ink:#172033; --fx-muted:#64748b; --fx-bg:#f7f8fb; }
          body { background: var(--fx-bg); color: var(--fx-ink); }
          .fx-shell { max-width: 1480px; margin: 0 auto; padding: 18px 22px 32px; }
          .fx-topbar { display:flex; align-items:center; justify-content:space-between; gap:18px; margin-bottom:14px; }
          .fx-title { font-size: 24px; font-weight: 720; letter-spacing: 0; }
          .fx-subtitle { color: var(--fx-muted); font-size: 13px; }
          .fx-band { background:white; border:1px solid var(--fx-border); border-radius:8px; padding:14px; }
          .fx-scenario-grid { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:10px; }
          .fx-scenario { min-height:154px; border:1px solid var(--fx-border); border-radius:8px; padding:12px; background:#fff; cursor:pointer; }
          .fx-scenario.active { border-color:#0f766e; box-shadow:0 0 0 2px rgba(15,118,110,.12); }
          .fx-scenario h3 { margin:0; font-size:16px; font-weight:700; }
          .fx-chip { display:inline-flex; align-items:center; gap:6px; padding:3px 8px; border-radius:999px; background:#eef2f7; font-size:12px; }
          .fx-main { display:grid; grid-template-columns: minmax(0,1fr) 420px; gap:12px; margin-top:12px; align-items:start; }
          .fx-pipeline { display:grid; grid-template-columns: repeat(9, minmax(112px,1fr)); gap:8px; align-items:center; overflow-x:auto; padding-bottom:4px; }
          .fx-node { min-height:92px; border:2px solid var(--c); border-radius:8px; background:white; padding:10px; text-align:left; cursor:pointer; }
          .fx-node.active { box-shadow:0 0 0 3px rgba(37,99,235,.16); }
          .fx-node-title { font-weight:700; font-size:13px; line-height:1.2; }
          .fx-node-status { margin-top:6px; color:var(--c); font-size:12px; font-weight:700; }
          .fx-edge-row { display:grid; grid-template-columns: repeat(8, minmax(112px,1fr)); gap:8px; margin:8px 60px 0 60px; overflow-x:auto; }
          .fx-edge { border-top:3px solid var(--c); color:var(--c); font-size:11px; padding-top:4px; white-space:nowrap; }
          .fx-detail-title { font-size:18px; font-weight:720; margin-bottom:4px; }
          .fx-section-title { font-size:14px; font-weight:720; margin:10px 0 6px; }
          .fx-formula { background:#f8fafc; border:1px solid var(--fx-border); border-radius:8px; padding:10px; overflow-x:auto; }
          .fx-reason { border-left:4px solid #0f766e; padding:8px 10px; background:#f0fdfa; border-radius:4px; }
          .fx-json { max-height:420px; overflow:auto; font-size:12px; }
          .q-table th { font-weight:700; }
          @media (max-width: 1100px) {
            .fx-scenario-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
            .fx-main { grid-template-columns: 1fr; }
          }
        </style>
        """
    )

    def selected_scenario() -> dict[str, Any]:
        return scenario_by_id[current["scenario_id"]]

    def selected_node() -> dict[str, Any]:
        scenario = selected_scenario()
        return next((n for n in scenario.get("pipeline", []) if n.get("node_id") == current["node_id"]), scenario["pipeline"][0])

    with ui.element("div").classes("fx-shell"):
        with ui.element("div").classes("fx-topbar"):
            with ui.element("div"):
                ui.label("FuzzyXAI Studio").classes("fx-title")
                ui.label("Кликабельный маршрут операторов: данные → проверка → диагностическое состояние → действие → отчёт").classes("fx-subtitle")
            with ui.row().classes("gap-2"):
                export_btn = ui.button("Export JSON", icon="download").props("outline")
                copy_btn = ui.button("Copy report summary", icon="content_copy").props("outline")

        scenarios_box = ui.element("div").classes("fx-scenario-grid")

        with ui.element("div").classes("fx-main"):
            left = ui.element("div").classes("fx-band")
            right = ui.element("div").classes("fx-band")

        with left:
            title_row = ui.row().classes("items-center justify-between w-full")
            with title_row:
                pipeline_title = ui.label().classes("text-lg font-bold")
                status_counts = ui.label().classes("fx-subtitle")
            summary_table = ui.table(
                columns=[{"name": "key", "label": "Показатель", "field": "key"}, {"name": "value", "label": "Значение", "field": "value"}],
                rows=[],
                row_key="key",
            ).classes("w-full")
            ui.separator().classes("my-3")
            pipeline_box = ui.element("div").classes("fx-pipeline")
            edge_box = ui.element("div").classes("fx-edge-row")
            ui.separator().classes("my-3")
            with ui.row().classes("w-full gap-3"):
                chart_one = ui.echart({}).classes("grow h-64")
                chart_two = ui.echart({}).classes("grow h-64")
            ui.separator().classes("my-3")
            report_title = ui.label("Report").classes("text-lg font-bold")
            report_reason = ui.html().classes("fx-reason")
            report_json = ui.code("", language="json").classes("w-full fx-json")

        with right:
            detail_title = ui.label().classes("fx-detail-title")
            detail_meta = ui.label().classes("fx-subtitle")
            detail_formula = ui.html().classes("fx-formula")
            detail_desc = ui.label().classes("text-sm")
            ui.label("Inputs").classes("fx-section-title")
            inputs_table = ui.table(
                columns=[{"name": "key", "label": "Поле", "field": "key"}, {"name": "value", "label": "Значение", "field": "value"}],
                rows=[],
                row_key="key",
            ).classes("w-full")
            ui.label("Computed values").classes("fx-section-title")
            computed_table = ui.table(
                columns=[{"name": "key", "label": "Поле", "field": "key"}, {"name": "value", "label": "Значение", "field": "value"}],
                rows=[],
                row_key="key",
            ).classes("w-full")
            ui.label("Output / diagnostics / effect").classes("fx-section-title")
            output_code = ui.code("", language="json").classes("w-full fx-json")

    def typeset() -> None:
        try:
            ui.run_javascript("if (window.MathJax) { MathJax.typesetPromise && MathJax.typesetPromise(); }")
        except AssertionError:
            pass

    def render_scenarios() -> None:
        scenarios_box.clear()
        with scenarios_box:
            for scenario in scenarios:
                active = scenario["scenario_id"] == current["scenario_id"]
                summary = scenario.get("summary", {})
                counts = []
                for key in ["objects_total", "accept", "lower_confidence", "block", "baseline_critical_misses", "fuzzyxai_critical_misses"]:
                    if key in summary:
                        counts.append(f"{key}={summary[key]}")
                with ui.element("div").classes("fx-scenario active" if active else "fx-scenario").on(
                    "click", lambda _e, sid=scenario["scenario_id"]: select_scenario(sid)
                ):
                    ui.label(scenario["scenario_name"]).classes("text-base font-bold")
                    ui.label(f"{scenario.get('domain')} · {scenario.get('data_type')}").classes("fx-subtitle")
                    ui.html(f"<span class='fx-chip'>{scenario.get('status')}</span>").classes("mt-2")
                    ui.label(scenario.get("description", "")).classes("text-sm mt-2")
                    ui.label(" · ".join(counts[:4])).classes("fx-subtitle mt-2")

    def render_pipeline() -> None:
        scenario = selected_scenario()
        report = build_report(scenario)
        pipeline_title.text = f"{scenario['scenario_name']} pipeline"
        status_counts.text = " · ".join(f"{k}: {v}" for k, v in _pipeline_status_counts(scenario).items())
        summary_table.rows = _scenario_summary_rows(scenario)
        summary_table.update()
        pipeline_box.clear()
        with pipeline_box:
            for node in scenario.get("pipeline", []):
                c = _status_color(str(node.get("status")))
                active = node.get("node_id") == current["node_id"]
                with ui.element("button").classes("fx-node active" if active else "fx-node").style(f"--c:{c}").on(
                    "click", lambda _e, nid=node["node_id"]: select_node(nid)
                ):
                    ui.html(f"<div class='fx-node-title'>{node.get('title')}</div><div class='fx-node-status'>{node.get('status')}</div>")
        edge_box.clear()
        with edge_box:
            for edge in scenario.get("edges", []):
                c = _status_color(str(edge.get("status")))
                label = f"{edge.get('operator_id')} · {edge.get('status')}"
                ui.html(f"<div class='fx-edge' style='--c:{c}'>{label} →</div>")
        chart_one.options.clear()
        chart_one.options.update(_action_chart_option(scenario.get("summary", {})))
        chart_one.update()
        chart_two.options.clear()
        chart_two.options.update(_miss_chart_option(scenario.get("summary", {})))
        chart_two.update()
        report_reason.content = f"<b>Action:</b> {report.get('final_action')}<br><b>Reason:</b> {report.get('action_reason')}"
        report_json.content = json.dumps(report, ensure_ascii=False, indent=2)

    def render_details() -> None:
        node = selected_node()
        op = node.get("operator", {})
        detail_title.text = str(op.get("operator_name") or node.get("title"))
        detail_meta.text = f"Источник: глава {op.get('source_chapter')} · operator_id={op.get('operator_id')} · status={node.get('status')}"
        formula = op.get("formula_latex", "")
        detail_formula.content = f"\\({formula}\\)"
        detail_desc.text = str(op.get("description", ""))
        inputs_table.rows = _rows(node.get("inputs", {}))
        computed_table.rows = _rows(node.get("computed", {}))
        inputs_table.update()
        computed_table.update()
        output_code.content = json.dumps(
            {
                "output": node.get("output", {}),
                "diagnostics": node.get("diagnostics", []),
                "effect_on_final_action": node.get("effect_on_final_action", ""),
            },
            ensure_ascii=False,
            indent=2,
        )

    def render_all() -> None:
        render_scenarios()
        render_pipeline()
        render_details()

    def select_scenario(scenario_id: str) -> None:
        current["scenario_id"] = scenario_id
        scenario = selected_scenario()
        current["node_id"] = scenario["pipeline"][0]["node_id"]
        render_all()

    def select_node(node_id: str) -> None:
        current["node_id"] = node_id
        render_pipeline()
        render_details()
        typeset()

    def export_current() -> None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        scenario = selected_scenario()
        report = build_report(scenario)
        path = REPORT_DIR / f"{report['run_id']}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        ui.download(path)

    def copy_summary() -> None:
        text = _report_summary(build_report(selected_scenario()))
        ui.run_javascript(f"navigator.clipboard.writeText({json.dumps(text)});")
        ui.notify("Report summary copied")

    export_btn.on_click(export_current)
    copy_btn.on_click(copy_summary)
    render_all()
    ui.timer(0.5, typeset, once=True)
    ui.run(title="FuzzyXAI Studio", port=port, reload=False, show=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8097)
    args = parser.parse_args()
    run(port=int(args.port))


if __name__ == "__main__":
    main()
