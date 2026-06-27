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
    STATUS_COLORS,
    STATUS_LABELS,
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
    return STATUS_COLORS.get(str(status), mapping.get(str(status), STATUS_COLOR.get(str(status), "#334155")))


def _status_label(status: str) -> str:
    return STATUS_LABELS.get(str(status), str(status).replace("_", " "))


NODE_TITLE_RU = {
    "Input Artifact": "Входной артефакт",
    "Adapter": "Адаптер",
    "Explanation Object Eₖ": "Объект Eₖ",
    "Interface Alignment Tᵢⱼ": "Согласование Tᵢⱼ",
    "Uncertainty Representation F": "Представление F",
    "Reduction Δ": "Редукция Δ",
    "Risk Observer": "Риск-наблюдатель",
    "Action": "Действие",
    "Report / Proof Package": "Доказательный пакет",
}

ACTION_LABELS_RU = {
    "accept": "ПРИНЯТО",
    "lower_confidence": "СНИЖЕНО ДОВЕРИЕ",
    "request_more_data": "ЗАПРОС ДАННЫХ",
    "defer_to_human": "ПЕРЕДАНО ЭКСПЕРТУ",
    "block": "БЛОКИРОВКА",
    "audit_report": "АУДИТОРСКИЙ ОТЧЁТ",
}

BAR_LABELS_RU = {
    "accept": "принято",
    "lower_confidence": "снижено доверие",
    "block": "заблокировано",
    "baseline": "базовый режим",
    "FuzzyXAI": "маршрут FuzzyXAI",
}


def _node_title(node: dict[str, Any]) -> str:
    return NODE_TITLE_RU.get(str(node.get("title", "")), str(node.get("title", "")))


def _action_label(action: Any) -> str:
    return ACTION_LABELS_RU.get(str(action), str(action).upper())


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


def _compact_bar_html(title: str, rows: list[tuple[str, int, str]]) -> str:
    max_value = max([value for _, value, _ in rows] or [1])
    bars = []
    for label, value, color in rows:
        width = 0 if max_value == 0 else max(3, round(value / max_value * 100))
        bars.append(
            f"<div class='fx-bar-row'><div class='fx-bar-label'>{label}</div>"
            f"<div class='fx-bar-track'><div class='fx-bar-fill' style='width:{width}%;background:{color}'></div></div>"
            f"<div class='fx-bar-value'>{value}</div></div>"
        )
    return f"<div class='fx-compact-chart'><div class='fx-compact-title'>{title}</div>{''.join(bars)}</div>"


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

    def scenario_operator_ids(scenario_id: str) -> list[str]:
        scenario = scenario_by_id.get(scenario_id, {})
        ids: list[str] = []
        for node in scenario.get("pipeline", []):
            op_id = node.get("operator", {}).get("operator_id")
            if op_id and op_id not in ids:
                ids.append(op_id)
        return ids

    def model_operator_ids(model: dict[str, Any]) -> list[str]:
        ids: list[str] = []
        for sid in model.get("usedIn", []):
            for op_id in scenario_operator_ids(sid):
                if op_id not in ids:
                    ids.append(op_id)
        return ids

    def operator_model_ids(operator_id: str) -> list[str]:
        out: list[str] = []
        operator = operators_by_id.get(operator_id, {})
        for model in ecosystem["models"]:
            if any(sid in operator.get("usedInScenarios", []) for sid in model.get("usedIn", [])):
                out.append(model["id"])
        return out

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
          .fx-hero { background:linear-gradient(135deg,#ffffff 0%,#ecfeff 100%); border:1px solid #b7ede7; border-radius:12px; padding:24px; margin:6px 0 14px; }
          .fx-hero h1 { font-size:34px; line-height:1.05; margin:0 0 8px; font-weight:820; letter-spacing:0; }
          .fx-hero-route { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-top:14px; color:#0f766e; font-weight:760; }
          .fx-hero-route span { background:white; border:1px solid #99f6e4; border-radius:999px; padding:5px 10px; }
          .fx-card { background:var(--fx-card); border:1px solid var(--fx-border); border-radius:8px; padding:14px; }
          .fx-metric { background:white; border:1px solid var(--fx-border); border-radius:10px; padding:14px; min-height:90px; }
          .fx-metric-value { font-size:28px; font-weight:840; color:#0f172a; line-height:1; }
          .fx-metric-label { margin-top:8px; color:var(--fx-muted); font-size:12px; font-weight:700; text-transform:uppercase; }
          .fx-main-finding { padding:14px 16px; border-radius:8px; background:#ecfeff; border:1px solid #99f6e4; color:#0f172a; font-size:14px; }
          .fx-action-banner { border-radius:12px; padding:16px 18px; margin:14px 0; border:1px solid; }
          .fx-action-banner.blocked { background:#fff1f2; border-color:#fecdd3; color:#881337; }
          .fx-action-label { font-size:12px; font-weight:850; text-transform:uppercase; letter-spacing:.06em; opacity:.76; }
          .fx-action-title { font-size:32px; font-weight:900; margin-top:3px; line-height:1; }
          .fx-action-reason { margin-top:8px; color:#3f1d24; font-size:14px; }
          .fx-grid-4 { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
          .fx-grid-3 { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
          .fx-grid-2 { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
          .fx-entity { border:1px solid var(--fx-border); border-radius:8px; padding:12px; background:white; cursor:pointer; min-height:158px; }
          .fx-entity.active { border-color:var(--fx-accent); box-shadow:0 0 0 2px rgba(15,118,110,.12); }
          .fx-entity-title { font-size:16px; font-weight:740; line-height:1.2; }
          .fx-badge { display:inline-flex; align-items:center; border-radius:999px; padding:3px 8px; font-size:12px; font-weight:650; background:#eef2f7; color:#334155; }
          .fx-badge.status { color:white; background:var(--c); }
          .fx-map { display:grid; grid-template-columns: repeat(6, minmax(120px,1fr)); gap:8px; align-items:stretch; }
          .fx-map-step { border:1px solid var(--fx-border); background:white; border-radius:8px; padding:12px; text-align:center; min-height:92px; }
          .fx-map-arrow { color:var(--fx-muted); text-align:center; margin-top:8px; }
          .fx-workspace { display:grid; grid-template-columns:270px minmax(0,1fr) 380px; gap:12px; align-items:start; }
          .fx-hybrid-header { background:white; border:1px solid var(--fx-border); border-radius:12px; padding:18px; margin:0 0 12px; }
          .fx-pipeline { display:grid; grid-template-columns:repeat(auto-fit,minmax(118px,1fr)); gap:10px; overflow:visible; padding:8px 2px 12px; align-items:stretch; }
          .fx-node { position:relative; min-height:108px; border:2px solid var(--c); background:white; border-radius:10px; padding:11px; cursor:pointer; text-align:left; }
          .fx-node.active { border:3px solid #111827 !important; box-shadow:0 0 0 4px rgba(17,24,39,.10); transform:translateY(-2px); }
          .fx-selected-pill { display:inline-block; margin-top:8px; padding:2px 8px; border-radius:999px; background:#111827; color:white; font-size:11px; font-weight:760; }
          .fx-node-index { width:24px; height:24px; border-radius:999px; display:flex; align-items:center; justify-content:center; background:#eef2f7; color:#334155; font-size:12px; font-weight:850; margin-bottom:8px; }
          .fx-node-title { font-weight:760; font-size:13px; line-height:1.18; }
          .fx-node-status { margin-top:7px; color:var(--c); font-size:12px; font-weight:760; }
          .fx-inspector-head { border:1px solid var(--fx-border); border-radius:10px; padding:12px; margin-bottom:10px; background:#f8fafc; }
          .fx-inspector-head.blocked, .fx-inspector-head.critical { background:#fff1f2; border-color:#fecdd3; color:#881337; }
          .fx-inspector-head.warning { background:#fffbeb; border-color:#fde68a; color:#713f12; }
          .fx-inspector-kicker { font-size:12px; font-weight:760; opacity:.78; }
          .fx-inspector-title { font-size:18px; font-weight:760; line-height:1.2; }
          .fx-section-title { font-size:14px; font-weight:760; margin:10px 0 6px; }
          .fx-human { border-left:4px solid var(--fx-accent); background:#f0fdfa; border-radius:4px; padding:9px 10px; font-size:14px; }
          .fx-diagnostic { border-left:4px solid #dc2626; background:#fef2f2; border-radius:4px; padding:9px 10px; font-size:13px; margin-top:8px; }
          .fx-link-list button { width:100%; justify-content:flex-start; }
          .fx-chip-list { display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }
          .fx-chip-list button { border:1px solid #cbd5e1; border-radius:999px; background:#f8fafc; padding:4px 8px; min-height:30px; font-size:12px; font-weight:700; text-transform:none !important; }
          .fx-chip-list .q-btn__content { text-transform:none !important; }
          .fx-compact-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; margin-top:10px; }
          .fx-compact-chart { border:1px solid var(--fx-border); border-radius:10px; padding:12px; background:#fff; }
          .fx-compact-title { font-weight:820; margin-bottom:10px; }
          .fx-bar-row { display:grid; grid-template-columns:118px minmax(80px,1fr) 42px; align-items:center; gap:8px; margin:8px 0; font-size:12px; }
          .fx-bar-label { color:#334155; font-weight:700; }
          .fx-bar-track { height:10px; border-radius:999px; background:#e2e8f0; overflow:hidden; }
          .fx-bar-fill { height:10px; border-radius:999px; }
          .fx-bar-value { text-align:right; font-weight:820; color:#0f172a; }
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
            ui.html(f"<span class='fx-badge status' style='--c:{c}'>{_status_label(entity.get('status', 'connected'))}</span>").classes("mt-2")
            ui.label(_short(entity.get("description", ""), 150)).classes("text-sm mt-2")
            tags = " ".join(f"<span class='fx-badge'>{t}</span>" for t in entity.get("tags", [])[:4])
            ui.html(tags).classes("mt-2")
            ui.button("Открыть", on_click=on_click).props("flat dense").classes("mt-2")

    def render_ecosystem() -> None:
        with content:
            with ui.element("div").classes("fx-hero"):
                ui.html("<h1>FuzzyXAI Studio</h1>")
                ui.label("Исследовательская экосистема для проверки объяснительных маршрутов").classes("text-base")
                with ui.element("div").classes("fx-hero-route"):
                    for part in ["Публикации", "Модели", "Операторы", "Сценарии", "Действия"]:
                        ui.html(f"<span>{part}</span>")
            with ui.element("div").classes("fx-grid-4 mt-3"):
                for title, value in [
                    ("Публикации", len(ecosystem["articles"])),
                    ("Модели", len(ecosystem["models"])),
                    ("Сценарии", len(ecosystem["scenarios"])),
                    ("Операторы", len(ecosystem["operators"])),
                ]:
                    with ui.element("div").classes("fx-metric"):
                        ui.html(f"<div class='fx-metric-value'>{value}</div><div class='fx-metric-label'>{title}</div>")
            with ui.element("div").classes("fx-card mt-3"):
                ui.label("Главный проверенный маршрут").classes("text-lg font-bold")
                with ui.element("div").classes("fx-main-finding mt-2"):
                    ui.html("<b>HYBRID-XIRIS:</b> статья / разработка → neuro-fuzzy iris matcher → Tᵢⱼ + NAS + риск-наблюдатель → <b>БЛОКИРОВКА</b>")
                ui.button("Открыть HYBRID-XIRIS workspace", on_click=lambda: set_view("scenario", scenario_id="hybrid_xiris", node_id="risk_observer")).props("unelevated color=teal").classes("mt-3")
            with ui.element("div").classes("fx-card mt-3"):
                ui.label("Карта маршрута HYBRID-XIRIS").classes("text-lg font-bold")
                with ui.element("div").classes("fx-map mt-2"):
                    for step, sub in [
                        ("Разработка", "HYBRID-XIRIS"),
                        ("Модель", "neuro-fuzzy iris matcher"),
                        ("Операторы", "Eₖ, Tᵢⱼ, NAS, риск"),
                        ("Сценарий", "контрольный прогон"),
                        ("Диагностика", "конфликт источников"),
                        ("Действие", "БЛОКИРОВКА"),
                    ]:
                        with ui.element("div").classes("fx-map-step"):
                            ui.label(step).classes("font-bold")
                            ui.label(sub).classes("fx-muted")
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
                    ui.html(f"<span class='fx-badge status' style='--c:{_color(model.get('status'))}'>{_status_label(model.get('status', 'connected'))}</span>").classes("mt-2")
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
                    ui.label("Операторы").classes("fx-section-title")
                    with ui.element("div").classes("fx-link-list"):
                        for op_id in model_operator_ids(model):
                            op = operators_by_id.get(op_id)
                            if op:
                                ui.button(op["title"], on_click=lambda _e=None, x=op_id: set_view("operator", operator_id=x)).props("flat dense")

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
        summary = scenario.get("summary", {})
        with content:
            with ui.element("div").classes("fx-hybrid-header"):
                ui.label(scenario["scenario_name"]).classes("text-2xl font-bold")
                ui.label("Биометрический сценарий: радужная оболочка" if scenario["scenario_id"] == "hybrid_xiris" else scenario.get("description", "")).classes("fx-muted")
                if scenario["scenario_id"] == "hybrid_xiris":
                    with ui.element("div").classes("fx-grid-4 mt-3"):
                        for label, value in [
                            ("Объектов", summary.get("objects_total", "—")),
                            ("Принято", summary.get("accept", "—")),
                            ("Снижено", summary.get("lower_confidence", "—")),
                            ("Заблокировано", summary.get("block", "—")),
                        ]:
                            with ui.element("div").classes("fx-metric"):
                                ui.html(f"<div class='fx-metric-value'>{value}</div><div class='fx-metric-label'>{label}</div>")
                    with ui.element("div").classes("fx-action-banner blocked"):
                        ui.html(
                            "<div class='fx-action-label'>Итоговое действие</div>"
                            f"<div class='fx-action-title'>{_action_label(report.get('final_action'))}</div>"
                            f"<div class='fx-action-reason'>{report.get('action_reason')}<br><b>код действия:</b> {report.get('final_action')}</div>"
                        )
                    with ui.element("div").classes("fx-main-finding mt-3"):
                        ui.html(
                            f"<b>Главный обнаруженный эффект:</b> модель поддерживает принятие, но низкое качество сегментации создаёт критический разрыв. "
                            f"Критические пропуски: базовый режим — {summary.get('baseline_critical_misses', '—')}, маршрут FuzzyXAI — {summary.get('fuzzyxai_critical_misses', '—')}."
                        )
                else:
                    ui.label(scenario.get("description", "")).classes("mt-1")
            with ui.element("div").classes("fx-workspace mt-3"):
                with ui.element("div").classes("fx-card"):
                    ui.label("Вход / модель").classes("text-lg font-bold")
                    if scenario["scenario_id"] == "hybrid_xiris":
                        ui.label("Neuro-fuzzy iris matcher").classes("font-bold mt-2")
                        ui.label("score совпадения + активированные правила").classes("fx-muted")
                        ui.label("Контрольный кейс").classes("fx-section-title")
                        for label, value in [
                            ("image_quality", "0.31"),
                            ("segmentation_quality", "0.27"),
                            ("model_match_signal", "0.88"),
                            ("alpha_accept", "0.82"),
                            ("alpha_block", "0.91"),
                        ]:
                            ui.html(f"<div class='fx-badge' style='margin:3px 3px 3px 0'>{label}: <b>{value}</b></div>")
                        ui.label("Связанные модели").classes("fx-section-title")
                        with ui.element("div").classes("fx-chip-list"):
                            for mid in ["model_iris_quality", "model_iris_matcher"]:
                                m = models_by_id.get(mid)
                                if m:
                                    label = m["title"].replace(" model", "").replace("Model", "").strip()
                                    ui.button(f"Модель · {label}", on_click=lambda _e=None, x=mid: set_view("model", model_id=x)).props("flat dense")
                    else:
                        ui.table(
                            columns=[{"name": "metric", "label": "Показатель", "field": "metric"}, {"name": "value", "label": "Значение", "field": "value"}],
                            rows=_metric_rows(summary),
                            row_key="metric",
                        ).classes("w-full")
                with ui.element("div").classes("fx-card"):
                    ui.label("Маршрут операторов").classes("text-lg font-bold")
                    if scenario["scenario_id"] == "hybrid_xiris":
                        ui.label("Вход → качество сегментации → сигнал модели → Eₖ → Tᵢⱼ → NAS → риск-наблюдатель → BLOCK").classes("fx-muted")
                    with ui.element("div").classes("fx-pipeline mt-2"):
                        for idx, n in enumerate(scenario["pipeline"], start=1):
                            c = _color(n.get("status"))
                            active = n["node_id"] == state["node_id"]
                            with ui.element("button").classes("fx-node active" if active else "fx-node").style(f"--c:{c}").on("click", lambda _e, nid=n["node_id"]: set_view("scenario", scenario_id=scenario["scenario_id"], node_id=nid)):
                                selected = "<div class='fx-selected-pill'>выбран</div>" if active else ""
                                ui.html(f"<div class='fx-node-index'>{idx}</div><div class='fx-node-title'>{_node_title(n)}</div><div class='fx-node-status'>{_status_label(n.get('status', 'info'))}</div>{selected}")
                    if scenario["scenario_id"] == "hybrid_xiris":
                        ui.html(
                            "<div class='fx-compact-grid'>"
                            + _compact_bar_html(
                                "Распределение действий",
                                [
                                    ("принято", int(summary.get("accept", 0)), "#0f766e"),
                                    ("снижено доверие", int(summary.get("lower_confidence", 0)), "#d97706"),
                                    ("заблокировано", int(summary.get("block", 0)), "#dc2626"),
                                ],
                            )
                            + _compact_bar_html(
                                "Критические пропуски",
                                [
                                    ("базовый режим", int(summary.get("baseline_critical_misses", 0)), "#b45309"),
                                    ("маршрут FuzzyXAI", int(summary.get("fuzzyxai_critical_misses", 0)), "#0f766e"),
                                ],
                            )
                            + "</div>"
                        )
                    else:
                        with ui.element("div").classes("fx-grid-2 mt-3"):
                            ui.echart(_scenario_action_chart(scenario.get("summary", {}))).classes("h-60")
                            ui.echart(_scenario_evidence_chart(scenario.get("summary", {}))).classes("h-60")
                    with ui.element("div").classes("fx-human mt-3"):
                        ui.html(f"<b>Итог:</b> {_action_label(report.get('final_action'))} <span class='fx-muted'>(код: {report.get('final_action')})</span><br><b>Почему:</b> {report.get('action_reason')}")
                render_inspector(node, scenario)

    def render_inspector(node: dict[str, Any], scenario: dict[str, Any]) -> None:
        op = node.get("operator", {})
        with ui.element("div").classes("fx-card"):
            status = node.get("status", "info")
            with ui.element("div").classes(f"fx-inspector-head {status}"):
                ui.html(f"<div class='fx-inspector-kicker'>{op.get('operator_name')} · глава {op.get('source_chapter')}</div>")
                ui.html(f"<div class='fx-inspector-title'>{_node_title(node)}</div>")
                ui.html(f"<span class='fx-badge status' style='--c:{_color(status)}'>{_status_label(status)}</span>").classes("mt-2")
            ui.label("Что проверяет").classes("fx-section-title")
            ui.label(op.get("description", "")).classes("text-sm")
            ui.label("Что произошло").classes("fx-section-title")
            ui.html(f"<div class='fx-human'>{node.get('effect_on_final_action', 'Проверяет участок объяснительного маршрута.')}</div>")
            if node.get("output"):
                ui.label("Итог блока").classes("fx-section-title")
                for key, value in node.get("output", {}).items():
                    ui.html(f"<span class='fx-badge'>{key}: <b>{_fmt(value)}</b></span>").classes("mr-1")
            diagnostics = node.get("diagnostics", [])
            if diagnostics:
                ui.label("Диагностика").classes("fx-section-title")
                for d in diagnostics:
                    ui.html(f"<div class='fx-diagnostic'><b>{d.get('type', 'diagnostic')}</b>: {d.get('reason', '')}<br>действие: <b>{d.get('recommended_action', '')}</b></div>")
            with ui.expansion("Входные сигналы", icon="input").classes("w-full mt-2"):
                ui.table(
                    columns=[{"name": "metric", "label": "Сигнал", "field": "metric"}, {"name": "value", "label": "Значение", "field": "value"}],
                    rows=_metric_rows(node.get("inputs", {})),
                    row_key="metric",
                ).classes("w-full")
            with ui.expansion("Что вычислено", icon="calculate").classes("w-full"):
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
                    with ui.element("div").classes("fx-link-list"):
                        for sid in sorted(set(operator.get("usedInScenarios", []))):
                            s = scenario_by_id.get(sid)
                            if s:
                                ui.button(s["scenario_name"], on_click=lambda _e=None, x=sid: set_view("scenario", scenario_id=x, node_id="alignment")).props("flat dense")
                    ui.label("Связанные модели").classes("fx-section-title")
                    with ui.element("div").classes("fx-link-list"):
                        for mid in operator_model_ids(operator["id"]):
                            m = models_by_id.get(mid)
                            if m:
                                ui.button(m["title"], on_click=lambda _e=None, x=mid: set_view("model", model_id=x)).props("flat dense")
                    ui.label("Типовые исходы").classes("fx-section-title")
                    for out in operator.get("possibleOutcomes", []):
                        ui.html(f"<span class='fx-badge'>{out}</span>").classes("mr-1")
                with ui.element("div").classes("fx-card"):
                    ui.label("Реальные срабатывания").classes("text-lg font-bold")
                    for case in operator.get("realCases", [])[:8]:
                        with ui.element("div").classes("fx-card mt-2"):
                            ui.label(case.get("scenario_name", "")).classes("font-bold")
                            ui.label(f"{NODE_TITLE_RU.get(str(case.get('node_title')), case.get('node_title'))} · {_status_label(case.get('status', 'info'))}").classes("fx-muted")
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
