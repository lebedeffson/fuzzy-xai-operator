"""FuzzyXAI Studio: ecosystem workspace, not a debug dashboard."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fuzzyxai.studio.operator_scenarios import (  # noqa: E402
    STATUS_COLOR,
    build_ecosystem_entities,
    build_report,
    ensure_scenario_json_files,
    load_scenarios,
)

REPORT_DIR = ROOT / "reports" / "studio"


def _color(status: str) -> str:
    mapping = {
        "draft": "#64748b",
        "connected": "#2563eb",
        "verified": "#16a34a",
        "limited": "#d97706",
        "blocked": "#dc2626",
        "passed": "#16a34a",
        "warning": "#d97706",
        "critical": "#dc2626",
        "info": "#2563eb",
        "source-pending": "#d97706",
        "fixture-certified": "#16a34a",
        "scenario_run_verified": "#16a34a",
    }
    return mapping.get(str(status), STATUS_COLOR.get(str(status), "#334155"))


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _metric_rows(mapping: dict[str, Any]) -> list[dict[str, str]]:
    return [{"metric": str(k), "value": _fmt(v)} for k, v in (mapping or {}).items()]


def _short(text: str, limit: int = 120) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _scenario_action_chart(summary: dict[str, Any]) -> dict[str, Any]:
    labels = [k for k in ["accept", "lower_confidence", "request_more_data", "defer_to_human", "block"] if k in summary]
    return {
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 48, "right": 18, "top": 18, "bottom": 56},
        "xAxis": {"type": "category", "data": labels, "axisLabel": {"interval": 0, "rotate": 14}},
        "yAxis": {"type": "value"},
        "series": [{"type": "bar", "data": [summary.get(k, 0) for k in labels], "itemStyle": {"color": "#0f766e"}, "label": {"show": True, "position": "top"}}],
    }


def _scenario_evidence_chart(summary: dict[str, Any]) -> dict[str, Any]:
    if "baseline_critical_misses" in summary:
        labels = ["baseline misses", "FuzzyXAI misses"]
        data = [summary.get("baseline_critical_misses", 0), summary.get("fuzzyxai_critical_misses", 0)]
    elif "checks_without_beacon" in summary:
        labels = ["without BEACON", "with BEACON", "audit reports"]
        data = [summary.get("checks_without_beacon", 0), summary.get("checks_with_beacon", 0), summary.get("audit_reports", 0)]
    elif {"probability", "mean_alpha_k", "positive_feature_support"}.issubset(summary):
        labels = ["probability", "mean_alpha", "feature_support"]
        data = [summary.get("probability", 0), summary.get("mean_alpha_k", 0), summary.get("positive_feature_support", 0)]
    else:
        labels = [k for k, v in summary.items() if isinstance(v, (int, float))]
        data = [summary[k] for k in labels]
    return {
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 54, "right": 18, "top": 18, "bottom": 64},
        "xAxis": {"type": "category", "data": labels, "axisLabel": {"interval": 0, "rotate": 16}},
        "yAxis": {"type": "value"},
        "series": [{"type": "bar", "data": data, "itemStyle": {"color": "#b45309"}, "label": {"show": True, "position": "top"}}],
    }


def run(port: int = 8097) -> None:
    try:
        from nicegui import app, ui
    except Exception as exc:  # pragma: no cover
        raise SystemExit("FuzzyXAI Studio requires nicegui. Install requirements.txt first.") from exc

    ensure_scenario_json_files()
    scenarios = load_scenarios()
    ecosystem = build_ecosystem_entities(scenarios)
    scenario_by_id = {s["scenario_id"]: s for s in scenarios}
    models_by_id = {m["id"]: m for m in ecosystem["models"]}
    articles_by_id = {a["id"]: a for a in ecosystem["articles"]}
    operators_by_id = {o["id"]: o for o in ecosystem["operators"]}

    state = {
        "view": "ecosystem",
        "scenario_id": "hybrid_xiris" if "hybrid_xiris" in scenario_by_id else scenarios[0]["scenario_id"],
        "node_id": "input_artifact",
        "article_id": ecosystem["articles"][0]["id"],
        "model_id": ecosystem["models"][0]["id"],
        "operator_id": "build_Ek" if "build_Ek" in operators_by_id else ecosystem["operators"][0]["id"],
    }

    app.add_static_files("/studio_reports", str(REPORT_DIR))
    ui.add_head_html(
        """
        <script>
          window.MathJax = {tex: {inlineMath: [['\\\\(','\\\\)'], ['$', '$']]}};
        </script>
        <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <style>
          :root { --fx-bg:#f6f7fb; --fx-card:#ffffff; --fx-border:#d8dee9; --fx-ink:#172033; --fx-muted:#637083; --fx-accent:#0f766e; }
          body { background:var(--fx-bg); color:var(--fx-ink); }
          .fx-shell { max-width:1500px; margin:0 auto; padding:18px 22px 34px; }
          .fx-top { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:12px; }
          .fx-title { font-size:24px; font-weight:760; letter-spacing:0; }
          .fx-muted { color:var(--fx-muted); font-size:13px; }
          .fx-nav { display:flex; gap:6px; flex-wrap:wrap; }
          .fx-nav button { border:1px solid var(--fx-border); background:white; border-radius:8px; padding:8px 12px; font-weight:650; }
          .fx-nav button.active { background:#0f766e; color:white; border-color:#0f766e; }
          .fx-card { background:var(--fx-card); border:1px solid var(--fx-border); border-radius:8px; padding:14px; }
          .fx-grid-4 { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
          .fx-grid-3 { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
          .fx-grid-2 { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
          .fx-entity { border:1px solid var(--fx-border); border-radius:8px; padding:12px; background:white; cursor:pointer; min-height:150px; }
          .fx-entity.active { border-color:var(--fx-accent); box-shadow:0 0 0 2px rgba(15,118,110,.12); }
          .fx-entity-title { font-size:16px; font-weight:740; line-height:1.2; }
          .fx-badge { display:inline-flex; align-items:center; border-radius:999px; padding:3px 8px; font-size:12px; font-weight:650; background:#eef2f7; color:#334155; }
          .fx-badge.status { color:white; background:var(--c); }
          .fx-map { display:grid; grid-template-columns: repeat(6, minmax(120px,1fr)); gap:8px; align-items:stretch; }
          .fx-map-step { border:1px solid var(--fx-border); background:white; border-radius:8px; padding:12px; text-align:center; min-height:92px; }
          .fx-map-arrow { color:var(--fx-muted); text-align:center; margin-top:8px; }
          .fx-workspace { display:grid; grid-template-columns:280px minmax(0,1fr) 390px; gap:12px; align-items:start; }
          .fx-pipeline { display:flex; gap:8px; overflow-x:auto; padding-bottom:4px; }
          .fx-node { min-width:136px; min-height:96px; border:2px solid var(--c); background:white; border-radius:8px; padding:10px; cursor:pointer; text-align:left; }
          .fx-node.active { box-shadow:0 0 0 3px rgba(37,99,235,.16); }
          .fx-node-title { font-weight:760; font-size:13px; line-height:1.18; }
          .fx-node-status { margin-top:7px; color:var(--c); font-size:12px; font-weight:760; }
          .fx-edge-strip { display:flex; gap:8px; overflow-x:auto; margin:8px 20px 0; }
          .fx-edge { min-width:128px; border-top:3px solid var(--c); color:var(--c); padding-top:4px; font-size:11px; white-space:nowrap; }
          .fx-inspector-title { font-size:18px; font-weight:760; line-height:1.2; }
          .fx-section-title { font-size:14px; font-weight:760; margin:10px 0 6px; }
          .fx-human { border-left:4px solid var(--fx-accent); background:#f0fdfa; border-radius:4px; padding:9px 10px; font-size:14px; }
          .fx-formula { background:#f8fafc; border:1px solid var(--fx-border); border-radius:8px; padding:10px; overflow-x:auto; }
          .fx-technical { max-height:360px; overflow:auto; font-size:12px; }
          .fx-action { font-size:24px; font-weight:800; }
          @media (max-width:1180px) { .fx-grid-4 { grid-template-columns:repeat(2,minmax(0,1fr)); } .fx-workspace { grid-template-columns:1fr; } }
        </style>
        """
    )

    def current_scenario() -> dict[str, Any]:
        return scenario_by_id[state["scenario_id"]]

    def current_node() -> dict[str, Any]:
        scenario = current_scenario()
        return next((n for n in scenario["pipeline"] if n["node_id"] == state["node_id"]), scenario["pipeline"][0])

    with ui.element("div").classes("fx-shell"):
        with ui.element("div").classes("fx-top"):
            with ui.element("div"):
                ui.label("FuzzyXAI Studio").classes("fx-title")
                ui.label("Исследовательская рабочая станция: публикации → модели → операторы → сценарии → действие").classes("fx-muted")
            nav = ui.element("div").classes("fx-nav")
        content = ui.element("div")

    def typeset() -> None:
        try:
            ui.run_javascript("if (window.MathJax?.typesetPromise) MathJax.typesetPromise();")
        except AssertionError:
            pass

    def set_view(view: str, **kwargs: Any) -> None:
        state["view"] = view
        state.update(kwargs)
        render()

    def render_nav() -> None:
        nav.clear()
        items = [
            ("ecosystem", "Экосистема"),
            ("articles", "Публикации"),
            ("models", "Модели"),
            ("scenarios", "Сценарии"),
            ("operators", "Операторы"),
            ("runs", "Прогоны"),
        ]
        with nav:
            for key, label in items:
                b = ui.button(label).props("flat")
                if state["view"] == key:
                    b.classes("active")
                b.on_click(lambda _e=None, k=key: set_view(k))

    def entity_card(entity: dict[str, Any], on_click, active: bool = False) -> None:
        c = _color(entity.get("status", "connected"))
        with ui.element("div").classes("fx-entity active" if active else "fx-entity").on("click", on_click):
            ui.html(f"<div class='fx-entity-title'>{entity.get('title')}</div>")
            ui.label(entity.get("subtitle", "")).classes("fx-muted")
            ui.html(f"<span class='fx-badge status' style='--c:{c}'>{entity.get('status')}</span>").classes("mt-2")
            ui.label(_short(entity.get("description", ""), 150)).classes("text-sm mt-2")
            tags = " ".join(f"<span class='fx-badge'>{t}</span>" for t in entity.get("tags", [])[:4])
            ui.html(tags).classes("mt-2")
            ui.button("Открыть", on_click=on_click).props("flat dense").classes("mt-2")

    def render_ecosystem() -> None:
        with content:
            ui.label("Экосистема FuzzyXAI").classes("text-2xl font-bold")
            ui.label("Не dashboard с JSON, а карта связей: статья → модель → адаптер → оператор → сценарий → действие.").classes("fx-muted")
            with ui.element("div").classes("fx-grid-4 mt-3"):
                for title, value in [
                    ("Публикации", len(ecosystem["articles"])),
                    ("Модели", len(ecosystem["models"])),
                    ("Сценарии", len(ecosystem["scenarios"])),
                    ("Операторы", len(ecosystem["operators"])),
                ]:
                    with ui.element("div").classes("fx-card"):
                        ui.label(str(value)).classes("text-3xl font-bold")
                        ui.label(title).classes("fx-muted")
            with ui.element("div").classes("fx-card mt-3"):
                ui.label("Общая карта связей").classes("text-lg font-bold")
                with ui.element("div").classes("fx-map mt-2"):
                    for step, sub in [
                        ("Публикация", "что подключено"),
                        ("Модель", "вход / выход / адаптер"),
                        ("Адаптер", "каналы Eₖ"),
                        ("Оператор", "проверка и формула"),
                        ("Сценарий", "маршрут"),
                        ("Действие", "accept / audit / block"),
                    ]:
                        with ui.element("div").classes("fx-map-step"):
                            ui.label(step).classes("font-bold")
                            ui.label(sub).classes("fx-muted")
                            if step != "Действие":
                                ui.label("↓").classes("fx-map-arrow")
            with ui.element("div").classes("fx-grid-3 mt-3"):
                entity_card(ecosystem["articles"][0], lambda _e: set_view("article", article_id=ecosystem["articles"][0]["id"]))
                entity_card(models_by_id["model_iris_matcher"], lambda _e: set_view("model", model_id="model_iris_matcher"))
                entity_card({"title": "HYBRID-XIRIS → source conflict → BLOCK", "subtitle": "последний проверенный маршрут", "description": "Высокий model score конфликтует с низким качеством сегментации; риск-наблюдатель блокирует авто-принятие.", "tags": ["block", "chi_R_crit", "NAS"], "status": "verified"}, lambda _e: set_view("scenario", scenario_id="hybrid_xiris", node_id="risk_observer"))

    def render_articles() -> None:
        with content:
            ui.label("Публикации и разработки").classes("text-2xl font-bold")
            with ui.element("div").classes("fx-grid-3 mt-3"):
                for article in ecosystem["articles"]:
                    entity_card(article, lambda _e, aid=article["id"]: set_view("article", article_id=aid), article["id"] == state.get("article_id"))

    def render_article_page() -> None:
        article = articles_by_id[state.get("article_id", ecosystem["articles"][0]["id"])]
        with content:
            entity_card(article, lambda _e: None, True)
            with ui.element("div").classes("fx-grid-3 mt-3"):
                with ui.element("div").classes("fx-card"):
                    ui.label("Связанные модели").classes("font-bold")
                    for mid in article.get("connectedModels", []):
                        m = models_by_id.get(mid)
                        if m:
                            ui.button(m["title"], on_click=lambda _e=None, x=mid: set_view("model", model_id=x)).props("flat").classes("w-full justify-start")
                with ui.element("div").classes("fx-card"):
                    ui.label("Связанные операторы").classes("font-bold")
                    for oid in article.get("connectedOperators", []):
                        o = operators_by_id.get(oid)
                        if o:
                            ui.button(o["title"], on_click=lambda _e=None, x=oid: set_view("operator", operator_id=x)).props("flat").classes("w-full justify-start")
                with ui.element("div").classes("fx-card"):
                    ui.label("Связанные сценарии").classes("font-bold")
                    for sid in article.get("connectedScenarios", []):
                        s = scenario_by_id.get(sid)
                        if s:
                            ui.button(s["scenario_name"], on_click=lambda _e=None, x=sid: set_view("scenario", scenario_id=x, node_id="input_artifact")).props("flat").classes("w-full justify-start")

    def render_models() -> None:
        with content:
            ui.label("Модели и адаптеры").classes("text-2xl font-bold")
            with ui.element("div").classes("fx-grid-3 mt-3"):
                for model in ecosystem["models"]:
                    entity_card(model, lambda _e, mid=model["id"]: set_view("model", model_id=mid), model["id"] == state.get("model_id"))

    def render_model_page() -> None:
        model = models_by_id[state.get("model_id", ecosystem["models"][0]["id"])]
        with content:
            with ui.element("div").classes("fx-workspace"):
                with ui.element("div").classes("fx-card"):
                    ui.label(model["title"]).classes("text-xl font-bold")
                    ui.label(model.get("description", "")).classes("text-sm mt-2")
                    ui.html(f"<span class='fx-badge status' style='--c:{_color(model.get('status'))}'>{model.get('status')}</span>").classes("mt-2")
                with ui.element("div").classes("fx-card"):
                    ui.label("Карточка модели").classes("text-lg font-bold")
                    ui.table(
                        columns=[{"name": "metric", "label": "Поле", "field": "metric"}, {"name": "value", "label": "Значение", "field": "value"}],
                        rows=_metric_rows({k: model.get(k, "") for k in ["domain", "inputType", "outputType", "adapterId"]}),
                        row_key="metric",
                    ).classes("w-full")
                    ui.label("Что адаптер извлекает").classes("fx-section-title")
                    for ch in model.get("explainabilityChannels", []):
                        ui.html(f"<span class='fx-badge'>✓ {ch}</span>").classes("mr-1")
                with ui.element("div").classes("fx-card"):
                    ui.label("Статус совместимости").classes("text-lg font-bold")
                    for line in ["Карточка модуля: заполнена", "Адаптер: есть", "Trace: есть", "Диагностические состояния: поддерживаются", "Автоматическое действие: только после риск-наблюдателя"]:
                        ui.label("✓ " + line).classes("text-sm")
                    ui.label("Где используется").classes("fx-section-title")
                    for sid in model.get("usedIn", []):
                        s = scenario_by_id.get(sid)
                        if s:
                            ui.button(s["scenario_name"], on_click=lambda _e=None, x=sid: set_view("scenario", scenario_id=x, node_id="input_artifact")).props("outline")

    def render_scenarios() -> None:
        with content:
            ui.label("Сценарии").classes("text-2xl font-bold")
            with ui.element("div").classes("fx-grid-4 mt-3"):
                for scenario in scenarios:
                    entity_card({"title": scenario["scenario_name"], "subtitle": scenario["data_type"], "description": scenario["description"], "tags": [scenario["domain"], scenario["data_type"]], "status": scenario["status"]}, lambda _e, sid=scenario["scenario_id"]: set_view("scenario", scenario_id=sid, node_id="input_artifact"), scenario["scenario_id"] == state.get("scenario_id"))

    def render_scenario_workspace() -> None:
        scenario = current_scenario()
        node = current_node()
        report = build_report(scenario)
        with content:
            ui.label(scenario["scenario_name"]).classes("text-2xl font-bold")
            ui.label(scenario.get("description", "")).classes("fx-muted")
            with ui.element("div").classes("fx-workspace mt-3"):
                with ui.element("div").classes("fx-card"):
                    ui.label("Данные и модель").classes("text-lg font-bold")
                    ui.table(
                        columns=[{"name": "metric", "label": "Показатель", "field": "metric"}, {"name": "value", "label": "Значение", "field": "value"}],
                        rows=_metric_rows(scenario.get("summary", {})),
                        row_key="metric",
                    ).classes("w-full")
                with ui.element("div").classes("fx-card"):
                    ui.label("Маршрут операторов").classes("text-lg font-bold")
                    with ui.element("div").classes("fx-pipeline mt-2"):
                        for n in scenario["pipeline"]:
                            c = _color(n.get("status"))
                            active = n["node_id"] == state["node_id"]
                            with ui.element("button").classes("fx-node active" if active else "fx-node").style(f"--c:{c}").on("click", lambda _e, nid=n["node_id"]: set_view("scenario", scenario_id=scenario["scenario_id"], node_id=nid)):
                                ui.html(f"<div class='fx-node-title'>{n['title']}</div><div class='fx-node-status'>{n['status']}</div>")
                    with ui.element("div").classes("fx-edge-strip"):
                        for e in scenario.get("edges", []):
                            ui.html(f"<div class='fx-edge' style='--c:{_color(e.get('status'))}'>{e.get('operator_id')} · {e.get('status')} →</div>")
                    with ui.element("div").classes("fx-grid-2 mt-3"):
                        ui.echart(_scenario_action_chart(scenario.get("summary", {}))).classes("h-60")
                        ui.echart(_scenario_evidence_chart(scenario.get("summary", {}))).classes("h-60")
                    with ui.element("div").classes("fx-human mt-3"):
                        ui.html(f"<b>Итог:</b> {report.get('final_action')}<br><b>Почему:</b> {report.get('action_reason')}")
                render_inspector(node, scenario)

    def render_inspector(node: dict[str, Any], scenario: dict[str, Any]) -> None:
        op = node.get("operator", {})
        with ui.element("div").classes("fx-card"):
            ui.label(node.get("title", "")).classes("fx-inspector-title")
            ui.label(f"{op.get('operator_name')} · глава {op.get('source_chapter')}").classes("fx-muted")
            ui.label("Что проверяет").classes("fx-section-title")
            ui.label(op.get("description", "")).classes("text-sm")
            ui.label("Человеческое объяснение").classes("fx-section-title")
            ui.html(f"<div class='fx-human'>{node.get('effect_on_final_action', 'Проверяет участок объяснительного маршрута.')}</div>")
            ui.label("Входные сигналы").classes("fx-section-title")
            ui.table(
                columns=[{"name": "metric", "label": "Сигнал", "field": "metric"}, {"name": "value", "label": "Значение", "field": "value"}],
                rows=_metric_rows(node.get("inputs", {})),
                row_key="metric",
            ).classes("w-full")
            ui.label("Что вычислено").classes("fx-section-title")
            ui.table(
                columns=[{"name": "metric", "label": "Величина", "field": "metric"}, {"name": "value", "label": "Значение", "field": "value"}],
                rows=_metric_rows(node.get("computed", {})),
                row_key="metric",
            ).classes("w-full")
            with ui.expansion("Математическая проверка", icon="functions").classes("w-full mt-2"):
                ui.html(f"<div class='fx-formula'>\\({op.get('formula_latex', '')}\\)</div>")
            with ui.expansion("Технический след", icon="data_object").classes("w-full"):
                ui.code(json.dumps({"node": node, "report": build_report(scenario)}, ensure_ascii=False, indent=2), language="json").classes("fx-technical")
                with ui.row():
                    ui.button("Download trace", icon="download", on_click=lambda: download_trace(scenario)).props("outline")
                    ui.button("Copy machine-readable run", icon="content_copy", on_click=lambda: copy_trace(scenario)).props("outline")

    def render_operators() -> None:
        with content:
            ui.label("Операторы FuzzyXAI").classes("text-2xl font-bold")
            with ui.element("div").classes("fx-grid-3 mt-3"):
                for operator in ecosystem["operators"]:
                    entity_card(operator, lambda _e, oid=operator["id"]: set_view("operator", operator_id=oid), operator["id"] == state.get("operator_id"))

    def render_operator_page() -> None:
        operator = operators_by_id[state.get("operator_id", ecosystem["operators"][0]["id"])]
        with content:
            with ui.element("div").classes("fx-workspace"):
                with ui.element("div").classes("fx-card"):
                    ui.label(operator["title"]).classes("text-xl font-bold")
                    ui.label(operator.get("description", "")).classes("text-sm mt-2")
                    ui.label("Формула").classes("fx-section-title")
                    ui.html(f"<div class='fx-formula'>\\({operator.get('formula', '')}\\)</div>")
                with ui.element("div").classes("fx-card"):
                    ui.label("Где используется").classes("text-lg font-bold")
                    for sid in sorted(set(operator.get("usedInScenarios", []))):
                        s = scenario_by_id.get(sid)
                        if s:
                            ui.button(s["scenario_name"], on_click=lambda _e=None, x=sid: set_view("scenario", scenario_id=x, node_id="alignment")).props("flat").classes("w-full justify-start")
                    ui.label("Типовые исходы").classes("fx-section-title")
                    for out in operator.get("possibleOutcomes", []):
                        ui.html(f"<span class='fx-badge'>{out}</span>").classes("mr-1")
                with ui.element("div").classes("fx-card"):
                    ui.label("Реальные срабатывания").classes("text-lg font-bold")
                    for case in operator.get("realCases", [])[:6]:
                        with ui.element("div").classes("fx-card mt-2"):
                            ui.label(case.get("scenario_name", "")).classes("font-bold")
                            ui.label(f"{case.get('node_title')} · {case.get('status')}").classes("fx-muted")
                            ui.label(_short(case.get("effect", ""), 160)).classes("text-sm")

    def render_runs() -> None:
        with content:
            ui.label("Проверочные прогоны").classes("text-2xl font-bold")
            with ui.element("div").classes("fx-grid-3 mt-3"):
                for run in ecosystem["runs"]:
                    with ui.element("div").classes("fx-card"):
                        ui.label(run.get("run_id", "")).classes("font-bold")
                        ui.label(f"scenario={run.get('scenario_id')}").classes("fx-muted")
                        ui.html(f"<div class='fx-action'>{run.get('final_action')}</div>")
                        ui.label(run.get("action_reason", "")).classes("text-sm")
                        ui.button("Открыть сценарий", on_click=lambda _e=None, sid=run.get("scenario_id"): set_view("scenario", scenario_id=sid, node_id="risk_observer")).props("outline")

    def download_trace(scenario: dict[str, Any]) -> None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report = build_report(scenario)
        path = REPORT_DIR / f"{report['run_id']}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        ui.download(path)

    def copy_trace(scenario: dict[str, Any]) -> None:
        ui.run_javascript(f"navigator.clipboard.writeText({json.dumps(json.dumps(build_report(scenario), ensure_ascii=False, indent=2))});")
        ui.notify("Technical trace copied")

    def render() -> None:
        render_nav()
        content.clear()
        view = state["view"]
        with content:
            if view == "ecosystem":
                render_ecosystem()
            elif view == "articles":
                render_articles()
            elif view == "article":
                render_article_page()
            elif view == "models":
                render_models()
            elif view == "model":
                render_model_page()
            elif view == "scenarios":
                render_scenarios()
            elif view == "scenario":
                render_scenario_workspace()
            elif view == "operators":
                render_operators()
            elif view == "operator":
                render_operator_page()
            elif view == "runs":
                render_runs()

    render()
    ui.run(title="FuzzyXAI Studio", port=port, reload=False, show=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8097)
    args = parser.parse_args()
    run(port=int(args.port))


if __name__ == "__main__":
    main()
