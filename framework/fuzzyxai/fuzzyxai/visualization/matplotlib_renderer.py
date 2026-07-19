from __future__ import annotations

from pathlib import Path
from textwrap import shorten, wrap
from typing import Any, Mapping


_COLORS = {
    "supported": "#2f6b4f",
    "stable": "#2f6b4f",
    "limitation": "#b7791f",
    "conflict": "#a23b3b",
    "missing": "#7b8790",
    "unmeasured": "#7b8790",
}


def _finish(figure: Any, output_path: str | Path | None):
    figure.tight_layout()
    if output_path is None:
        return figure
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120, bbox_inches="tight", facecolor="white")
    return output


def _empty(title: str, message: str):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(16, 9))
    axis.axis("off")
    axis.set_title(title, loc="left", fontsize=20, fontweight="bold")
    axis.text(0.05, 0.78, message, fontsize=14, color="#58656e")
    return figure


def _story(spec: Mapping[str, Any]):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    figure, axis = plt.subplots(figsize=(16, 9))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0.03, 0.94, "Карта доказательного объяснения", fontsize=22, fontweight="bold", color="#17242d")
    level = spec.get("overview", {}).get("explanation_level", {})
    axis.text(0.03, 0.895, f"Уровень {level.get('level', 'E0')}: {level.get('rationale', '')}", fontsize=11, color="#58656e")
    stages = list(spec.get("story", []))
    width, gap, left = 0.17, 0.025, 0.025
    for index, stage in enumerate(stages):
        x = left + index * (width + gap)
        evidence_status = str(stage.get("evidence_status", "insufficient_evidence"))
        color = {
            "adverse": _COLORS["conflict"],
            "mixed": _COLORS["limitation"],
            "unknown": _COLORS["missing"],
        }.get(str(stage.get("effect")), _COLORS["supported"])
        edge_color = _COLORS["limitation"] if evidence_status != "supported" else color
        box = FancyBboxPatch((x, 0.28), width, 0.48, boxstyle="round,pad=0.012,rounding_size=0.012", facecolor="#f7f9fa", edgecolor=edge_color, linewidth=2)
        axis.add_patch(box)
        axis.text(x + 0.012, 0.70, stage.get("title", ""), fontsize=15, fontweight="bold", color="#17242d")
        axis.text(x + width - 0.012, 0.70, f"{evidence_status.upper()} / {str(stage.get('effect', '')).upper()}", ha="right", fontsize=6.8, color=edge_color, fontweight="bold")
        y = 0.64
        facts = list(stage.get("facts", [])) or ["Evidence отсутствует"]
        for fact in facts[:4]:
            lines = wrap(str(fact), 25)[:3]
            axis.text(x + 0.012, y, "\n".join(lines), fontsize=9.5, va="top", color="#25343d")
            y -= 0.095 + 0.02 * max(0, len(lines) - 1)
        refs = ", ".join(stage.get("claim_refs", [])[:5]) or "нет claims"
        axis.text(x + 0.012, 0.31, refs, fontsize=8, color="#66757e")
        if index < len(stages) - 1:
            axis.annotate("", xy=(x + width + gap - 0.004, 0.52), xytext=(x + width + 0.004, 0.52), arrowprops={"arrowstyle": "->", "color": "#355b72", "lw": 2})
    axis.text(0.03, 0.13, "Каждый текстовый вывод связан с claim ID; inspect раскрывает evidence и provenance.", fontsize=12, color="#355b72")
    return figure


def _data_profile(spec: Mapping[str, Any]):
    import matplotlib.pyplot as plt

    profiles = list(spec.get("data_profile", []))
    if not profiles:
        return _empty("Профиль данных", "Референсные данные не переданы; percentile-профиль недоступен.")
    shown = profiles[:10]
    figure, axis = plt.subplots(figsize=(16, 9))
    axis.set_xlim(0, 100)
    axis.set_ylim(-1, len(shown))
    axis.set_xlabel("Перцентиль относительно референса")
    axis.set_title("Положение объекта относительно референсного профиля", loc="left", fontsize=20, fontweight="bold")
    for index, item in enumerate(reversed(shown)):
        y = index
        axis.plot([5, 95], [y, y], color="#b8c4ca", lw=12, solid_capstyle="round")
        axis.plot([50, 50], [y - 0.18, y + 0.18], color="#17242d", lw=2)
        percentile = item.get("percentile")
        color = _COLORS["conflict"] if item.get("anomaly_status") == "deviation_not_error" else _COLORS["supported"]
        if percentile is not None:
            axis.scatter([percentile], [y], s=110, color=color, edgecolors="white", zorder=3)
        value = item.get("object_value")
        contribution = item.get("contribution")
        suffix = f"; вклад {contribution:+.3f}" if contribution is not None else "; вклад не измерен"
        axis.text(101, y, f"значение {value}{suffix}", va="center", fontsize=9, clip_on=False)
    axis.set_yticks(range(len(shown)), [item.get("feature", "") for item in reversed(shown)])
    axis.grid(axis="x", alpha=0.18)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.text(0, -0.8, "Серая полоса: 5-95-й перцентили; чёрная риска: медиана; точка: объясняемый объект.", fontsize=10, color="#58656e")
    return figure


def _training_trace(spec: Mapping[str, Any]):
    import matplotlib.pyplot as plt

    timelines = list(spec.get("training_timeline", []))
    if not timelines:
        return _empty("Траектория обучения", "История обучения недоступна: модель загружена без наблюдаемых checkpoints/epoch metrics.")
    timeline = timelines[0]
    points = list(timeline.get("points", []))
    epochs = [point.get("epoch") for point in points]
    figure, axes = plt.subplots(6, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [0.5, 1, 1, 1, 1, 1.2]})
    figure.suptitle(f"Траектория обучения объекта {timeline.get('object_id')}", x=0.06, ha="left", fontsize=20, fontweight="bold")
    correct = [1 if point.get("correct") else 0 for point in points]
    axes[0].scatter(epochs, correct, c=[_COLORS["supported"] if flag else _COLORS["conflict"] for flag in correct], s=65)
    axes[0].set_yticks([0, 1], ["ошибка", "верно"])
    axes[1].plot(epochs, [point.get("confidence") for point in points], marker="o", color="#286f91")
    axes[1].set_ylabel("Confidence")
    axes[2].plot(epochs, [point.get("loss") for point in points], marker="o", color="#b7791f")
    axes[2].set_ylabel("Object loss")
    axes[3].plot(epochs, [point.get("margin") for point in points], marker="o", color="#70558c")
    axes[3].set_ylabel("Margin")
    axes[4].plot(epochs, [point.get("prototype_distance") for point in points], marker="o", color="#5a7582", label="prototype distance")
    axes[4].plot(epochs, [point.get("subgroup_metric") for point in points], marker="s", color="#a23b3b", label="subgroup metric")
    axes[4].plot(epochs, [point.get("global_metric") for point in points], marker="^", color="#2f6b4f", label="global metric")
    axes[4].set_ylabel("Context")
    axes[4].legend(loc="upper right", ncol=3, fontsize=8)
    rule_ids = sorted({item.get("rule_id") for point in points for item in point.get("rule_activations", []) if item.get("rule_id")})[:5]
    for rule_id in rule_ids:
        axes[5].plot(epochs, [next((item.get("activation") for item in point.get("rule_activations", []) if item.get("rule_id") == rule_id), None) for point in points], marker=".", label=rule_id)
    axes[5].set_ylabel("Rule activation")
    axes[5].set_xlabel("Эпоха")
    if rule_ids:
        axes[5].legend(loc="upper right", ncol=min(3, len(rule_ids)))
    for point in points:
        if point.get("forgetting"):
            for axis in axes:
                axis.axvline(point.get("epoch"), color=_COLORS["conflict"], ls="--", alpha=0.7)
    for axis in axes:
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    return figure


def _knowledge_atlas(spec: Mapping[str, Any]):
    import matplotlib.pyplot as plt

    atlas = dict(spec.get("knowledge_atlas", {}))
    concepts, rules = list(atlas.get("concepts", [])), list(atlas.get("rules", []))
    if not concepts and not rules:
        return _empty("Атлас знаний модели", "Нативные правила, концепты или измеренный суррогат недоступны.")
    figure, (left, right) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.05, 1.4]})
    figure.suptitle("Атлас знаний модели", x=0.04, ha="left", fontsize=20, fontweight="bold")
    for axis in (left, right):
        axis.axis("off")
    left.set_title("Концепты классов", loc="left", fontsize=15, fontweight="bold")
    y = 0.93
    for concept in concepts[:7]:
        text = f"{concept.get('class_name')}\n{shorten(str(concept.get('description')), 78)}\ncoverage={concept.get('primary_rule_coverage')}; uncovered={concept.get('uncovered_fraction')}\nrules: {', '.join(concept.get('primary_rules', [])[:5]) or 'нет'}"
        left.text(0.02, y, text, va="top", fontsize=10.5, bbox={"boxstyle": "round,pad=.55", "facecolor": "#f5f8f9", "edgecolor": "#9fb2bc"})
        y -= 0.19
    right.set_title("Правила: importance и coverage показаны раздельно", loc="left", fontsize=15, fontweight="bold")
    y = 0.93
    for rule in rules[:10]:
        kind = "native" if rule.get("native") else "surrogate"
        text = f"{rule.get('rule_id')} [{kind}]  importance={rule.get('importance')}  coverage={rule.get('coverage')}\n{shorten(str(rule.get('text')), 100)}"
        right.text(0.02, y, text, va="top", fontsize=10, color="#25343d")
        y -= 0.09
    return figure


def _decision_evidence(spec: Mapping[str, Any]):
    import matplotlib.pyplot as plt

    decision = dict(spec.get("decision_evidence", {}))
    figure, axes = plt.subplots(1, 3, figsize=(16, 9))
    figure.suptitle("Карта evidence для решения", x=0.04, ha="left", fontsize=20, fontweight="bold")
    columns = [
        ("Поддерживает", decision.get("supports", []), _COLORS["supported"]),
        ("Противоречит", decision.get("contradicts", []), _COLORS["conflict"]),
        ("Ограничивает", decision.get("limitations", []), _COLORS["limitation"]),
    ]
    for axis, (title, items, color) in zip(axes, columns):
        axis.axis("off")
        axis.set_title(f"{title} ({len(items)})", loc="left", color=color, fontsize=15, fontweight="bold")
        y = 0.92
        for item in list(items)[:8]:
            text = f"{item.get('claim_id')}  {item.get('statement')}"
            axis.text(0.01, y, "\n".join(wrap(text, 43)[:3]), va="top", fontsize=10, bbox={"boxstyle": "round,pad=.4", "facecolor": "#fafbfb", "edgecolor": color})
            y -= 0.115
    return figure


def _table_view(spec: Mapping[str, Any], view: str):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(16, 9))
    axis.axis("off")
    if view == "similar_cases":
        items = list(spec.get("similar_cases", []))
        title = "Похожие случаи и смысл сходства"
        columns = ["Объект", "Score", "Метод", "Представление", "Ограничения"]
        rows = [[item.get("reference_object_id"), f"{item.get('score', 0):.3f}", item.get("method"), item.get("representation"), "; ".join(item.get("limitations", []))] for item in items[:12]]
    elif view == "rule_ablation":
        ablations = list(spec.get("rule_ablations", []))
        title = "What-if: измеренный эффект отключения правила"
        columns = ["Правило", "Метрика", "С правилом", "Без правила", "Разница", "Scope"]
        rows = [[item.get("rule_id"), item.get("metric"), item.get("with_rule"), item.get("without_rule"), item.get("difference"), item.get("scope")] for item in ablations]
    else:
        items = list(spec.get("counterfactuals", []))
        title = "Контрфактические изменения"
        columns = ["Было", "Стало", "Изменения", "Эффект", "Действуемость"]
        rows = [[item.get("source_prediction"), item.get("target_prediction"), str(item.get("changed_features") or item.get("changed_rules")), item.get("observed_effect"), item.get("actionability")] for item in items[:12]]
    axis.set_title(title, loc="left", fontsize=20, fontweight="bold")
    if not rows:
        axis.text(0.02, 0.84, "Измеренный evidence для этого представления отсутствует.", fontsize=13, color="#58656e")
        return figure
    table = axis.table(cellText=rows, colLabels=columns, loc="upper left", cellLoc="left", colLoc="left", bbox=[0, 0.2, 1, 0.68])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    return figure


def _provenance(spec: Mapping[str, Any]):
    import matplotlib.pyplot as plt

    nodes = list(spec.get("provenance_nodes", []))
    edges = list(spec.get("provenance_edges", []))
    figure, axis = plt.subplots(figsize=(16, 9))
    axis.axis("off")
    axis.set_title("Трассируемость: evidence → claim → diagnostic → action", loc="left", fontsize=20, fontweight="bold")
    by_type: dict[str, list[Mapping[str, Any]]] = {}
    for node in nodes:
        by_type.setdefault(str(node.get("node_type")), []).append(node)
    order = ["data", "anomaly", "training_event", "rule", "concept", "similar_case", "counterfactual", "prediction", "claim", "diagnostic", "action"]
    selected = [node for kind in order for node in by_type.get(kind, [])[:4]]
    positions = {}
    for index, node in enumerate(selected):
        x = 0.05 + 0.9 * index / max(len(selected) - 1, 1)
        y = 0.63 if index % 2 == 0 else 0.38
        positions[node.get("node_id")] = (x, y)
        axis.text(x, y, f"{node.get('node_type')}\n{shorten(str(node.get('label')), 28)}", ha="center", va="center", fontsize=8.5, bbox={"boxstyle": "round,pad=.35", "facecolor": "#f6f8f9", "edgecolor": "#355b72"})
    for edge in edges:
        source, target = positions.get(edge.get("source")), positions.get(edge.get("target"))
        if source and target:
            axis.annotate("", xy=target, xytext=source, arrowprops={"arrowstyle": "->", "color": "#9aa8af", "lw": 0.8})
    axis.text(0.03, 0.12, f"Показано {len(selected)} из {len(nodes)} узлов. Полный граф доступен через result.audit() и inspect(...).", fontsize=11, color="#58656e")
    return figure


def _audit(spec: Mapping[str, Any]):
    import matplotlib.pyplot as plt

    overview = dict(spec.get("overview", {}))
    audit = dict(spec.get("audit", {}))
    counts = dict(overview.get("claim_counts", {}))
    figure, axis = plt.subplots(figsize=(16, 9))
    axis.axis("off")
    axis.set_title("Аудит объяснения", loc="left", fontsize=20, fontweight="bold")
    lines = [
        f"Action: {overview.get('action')}",
        f"Explanation level: {overview.get('explanation_level', {}).get('level')}",
        f"Claims: {counts.get('total')}",
        f"Supported: {counts.get('supported')}",
        f"Contested: {counts.get('contested')}",
        f"Insufficient evidence: {counts.get('insufficient_evidence')}",
        f"Graph validation: {'PASS' if audit.get('graph_valid') else 'FAIL'}",
        f"Graph nodes: {audit.get('node_count')}",
        f"Graph edges: {audit.get('edge_count')}",
    ]
    axis.text(0.03, 0.88, "\n".join(lines), va="top", fontsize=14, family="monospace", bbox={"boxstyle": "round,pad=.8", "facecolor": "#f6f8f9", "edgecolor": "#9fb2bc"})
    return figure


def render_visual_spec(spec: Mapping[str, Any], *, view: str, output_path: str | Path | None = None):
    """Render one focused view from the canonical visual specification."""

    aliases = {"dashboard": "explanation_story", "class_atlas": "knowledge_atlas", "counterfactual": "counterfactuals"}
    view = aliases.get(view, view)
    renderers = {
        "explanation_story": _story,
        "data_profile": _data_profile,
        "training_trace": _training_trace,
        "knowledge_atlas": _knowledge_atlas,
        "decision_evidence": _decision_evidence,
        "similar_cases": lambda payload: _table_view(payload, "similar_cases"),
        "counterfactuals": lambda payload: _table_view(payload, "counterfactuals"),
        "rule_ablation": lambda payload: _table_view(payload, "rule_ablation"),
        "provenance": _provenance,
        "audit": _audit,
    }
    if view not in renderers:
        raise ValueError(f"unsupported visualization view: {view}")
    return _finish(renderers[view](spec), output_path)
