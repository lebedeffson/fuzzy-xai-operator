from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from textwrap import shorten, wrap
from typing import Any

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


_NODE_TYPE_LABELS_RU = {
    "dataset": "датасет",
    "preprocessor": "предобработка",
    "model_artifact": "модель",
    "data": "данные объекта",
    "anomaly": "отклонение",
    "training_event": "обучение",
    "rule": "правило",
    "concept": "концепт класса",
    "similar_case": "похожий случай",
    "counterfactual": "контрфакт",
    "model_internals": "внутреннее устройство модели",
    "attribution_map": "карта атрибуции",
    "contribution": "вклад признака",
    "prediction": "прогноз",
    "claim": "утверждение",
    "diagnostic": "диагностика",
    "trace": "след",
    "action": "действие",
}


def _focused_provenance_chain(nodes: list[Mapping[str, Any]], edges: list[Mapping[str, Any]], selector: str) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, int]]:
    """P18 item 10: the anchor's real directed ANCESTRY only — a backward
    walk that follows edge.source from edge.target == current frontier,
    transitively. This is a genuine provenance question ("where did this
    come from") and it is directional by construction: it never crosses
    into a shared ancestor's OTHER descendants (e.g. sibling similar-case
    nodes reached only via a common `data:*` node), which the previous
    bidirectional (both-direction) BFS did — for a `claim` selector it used
    to pull in every neighbor of any node along the chain, including
    unrelated training objects that merely share a data node with the
    claim. A pure backward walk cannot do that: once it reaches a root node
    with no incoming edges (e.g. the dataset), it simply has nowhere left
    to go, instead of fanning back out along that root's outgoing edges.
    Returns (nodes, edges, depth-from-anchor per node_id) for a left-to-
    right layout by real ancestry depth."""

    node_by_id = {str(node.get("node_id")): node for node in nodes}
    anchor_ids = {selector} if selector in node_by_id else {str(node.get("node_id")) for node in nodes if str(node.get("node_type")) == selector}
    if not anchor_ids:
        anchor_ids = {"action"} if "action" in node_by_id else set()
    depth = {anchor: 0 for anchor in anchor_ids}
    frontier = set(anchor_ids)
    while frontier:
        next_frontier: set[str] = set()
        for edge in edges:
            source, target = str(edge.get("source")), str(edge.get("target"))
            if target in frontier and source not in depth:
                depth[source] = depth[target] + 1
                next_frontier.add(source)
        frontier = next_frontier
    related_nodes = [node_by_id[node_id] for node_id in depth if node_id in node_by_id]
    related_edges = [edge for edge in edges if str(edge.get("source")) in depth and str(edge.get("target")) in depth]
    return related_nodes, related_edges, depth


def _provenance(spec: Mapping[str, Any], *, selector: str | None = None):
    """P17: the default view is a FOCUSED subgraph (5-10 nodes) around one
    claim or the final action — not an arbitrary sample of the full
    80+-node graph, which answers no specific question. The complete graph
    is still available in full via result.audit()/to_dict(detail="audit").
    """

    import matplotlib.pyplot as plt

    nodes = list(spec.get("provenance_nodes", []))
    edges = list(spec.get("provenance_edges", []))
    anchor = selector or "action"
    # P19 system actions have a canonical operator ancestry.  Render that
    # real graph subroute in semantic rows rather than allowing generic claim
    # ancestors to consume the focused-action budget.
    if anchor == "action" and any(str(node.get("node_id")) == "system:E_model" for node in nodes):
        ids = {
            "dataset:root", "model_artifact:root", "prediction", "system:E_model", "system:T_ij",
            "system:aligned_E_model", "system:E_target", "system:Gamma", "system:U_model",
            "system:U_rules", "system:U_trace", "system:u_M", "system:representation",
            "system:reduction", "system:Delta", "system:E_pre", "system:I_pre", "system:rho",
            "system:rho_p", "system:one_minus_I_pre", "system:chi_R", "system:threshold_policy",
            "system:candidate_action", "system:critical_override", "system:policy_resolution", "system:critical", "action",
        }
        selected = [node for node in nodes if str(node.get("node_id")) in ids]
        allowed = {
            ("dataset:root", "prediction"), ("model_artifact:root", "prediction"), ("prediction", "system:E_model"),
            ("system:E_model", "system:T_ij"), ("system:T_ij", "system:aligned_E_model"),
            ("system:aligned_E_model", "system:Gamma"), ("system:E_target", "system:Gamma"),
            ("system:U_model", "system:u_M"), ("system:U_rules", "system:u_M"), ("system:U_trace", "system:u_M"),
            ("system:u_M", "system:representation"), ("system:representation", "system:reduction"),
            ("system:reduction", "system:Delta"), ("system:Delta", "system:E_pre"),
            ("system:aligned_E_model", "system:E_pre"), ("system:E_target", "system:E_pre"), ("system:u_M", "system:E_pre"),
            ("system:E_pre", "system:I_pre"), ("prediction", "system:rho_p"),
            ("system:I_pre", "system:one_minus_I_pre"), ("system:Gamma", "system:chi_R"), ("system:U_trace", "system:chi_R"),
            ("system:rho_p", "system:rho"), ("system:u_M", "system:rho"), ("system:one_minus_I_pre", "system:rho"),
            ("system:Delta", "system:rho"), ("system:chi_R", "system:rho"),
            ("system:rho", "system:threshold_policy"), ("system:threshold_policy", "system:candidate_action"),
            ("system:chi_R", "system:critical_override"), ("system:critical", "system:chi_R"),
            ("system:critical", "system:critical_override"), ("system:candidate_action", "system:policy_resolution"),
            ("system:critical_override", "system:policy_resolution"), ("system:policy_resolution", "action"),
        }
        selected_edges = [edge for edge in edges if (str(edge.get("source")), str(edge.get("target"))) in allowed]
        positions = {
            "dataset:root": (.05, .88), "model_artifact:root": (.18, .88), "prediction": (.32, .88),
            "system:E_model": (.46, .88), "system:T_ij": (.60, .88), "system:aligned_E_model": (.74, .88), "system:E_target": (.89, .88),
            "system:Gamma": (.88, .72), "system:chi_R": (.94, .57),
            "system:U_model": (.10, .65), "system:U_rules": (.10, .55), "system:U_trace": (.10, .45),
            "system:u_M": (.25, .55), "system:representation": (.38, .55), "system:reduction": (.51, .55), "system:Delta": (.64, .55),
            "system:E_pre": (.32, .32), "system:I_pre": (.46, .32), "system:one_minus_I_pre": (.60, .32),
            "system:rho_p": (.60, .43), "system:rho": (.76, .39),
            "system:threshold_policy": (.76, .22), "system:candidate_action": (.88, .22),
            "system:critical": (.46, .09), "system:critical_override": (.61, .09),
            "system:policy_resolution": (.82, .09), "action": (.96, .09),
        }
        figure, axis = plt.subplots(figsize=(20, 11))
        axis.axis("off")
        axis.set_title("Системный маршрут действия", loc="left", fontsize=18, fontweight="bold")
        for edge in selected_edges:
            source, target = positions.get(str(edge.get("source"))), positions.get(str(edge.get("target")))
            if source and target:
                axis.annotate("", xy=target, xytext=source, arrowprops={"arrowstyle": "->", "color": "#55798a", "lw": 1.3})
        for node in selected:
            node_id = str(node.get("node_id"))
            if node_id not in positions:
                continue
            label = str(node.get("label"))
            axis.text(*positions[node_id], "\n".join(wrap(label, width=16)[:2]), ha="center", va="center", fontsize=9,
                      bbox={"boxstyle": "round,pad=.42", "facecolor": "#e8f3fa" if node_id.startswith("system:") else "#f6f8f9", "edgecolor": "#355b72"})
        axis.text(.02, .02, "Маршрут построен из направленных узлов ExplanationGraph; боковые claims намеренно не заменяют операторную родословную.", fontsize=9, color="#58656e")
        return figure
    selected, selected_edges, depth = _focused_provenance_chain(nodes, edges, anchor)
    if len(selected) > 12:
        # Still too large for a focused picture (an unusually connected
        # anchor) — keep only the nodes closest to the anchor.
        keep_ids = {node_id for node_id, _ in sorted(depth.items(), key=lambda item: item[1])[:12]}
        selected = [node for node in selected if str(node.get("node_id")) in keep_ids]
        selected_edges = [edge for edge in selected_edges if str(edge.get("source")) in keep_ids and str(edge.get("target")) in keep_ids]

    # P18 item 10: an anchor with many same-depth ancestors (e.g. many rule
    # claims all directly preceding a prediction) used to stack all of them
    # into one column, overlapping into an unreadable smear. Capping each
    # column to 4 and naming the rest keeps every box readable without
    # dropping the fact that more ancestors exist at that depth.
    truncated_counts: dict[int, int] = {}
    by_depth_pre: dict[int, list[Mapping[str, Any]]] = {}
    for node in selected:
        by_depth_pre.setdefault(depth.get(str(node.get("node_id")), 0), []).append(node)
    kept_ids: set[str] = set()
    for level, level_nodes in by_depth_pre.items():
        kept = level_nodes[:4]
        kept_ids.update(str(node.get("node_id")) for node in kept)
        if len(level_nodes) > 4:
            truncated_counts[level] = len(level_nodes) - 4
    selected = [node for node in selected if str(node.get("node_id")) in kept_ids]
    selected_edges = [edge for edge in selected_edges if str(edge.get("source")) in kept_ids and str(edge.get("target")) in kept_ids]

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.axis("off")
    # P18 item 10: never show the raw selector string in the title — a
    # node_id resolves to its own (already Russian) label, a bare
    # node_type resolves through _NODE_TYPE_LABELS_RU.
    anchor_node = next((node for node in selected if str(node.get("node_id")) == anchor), None)
    anchor_display = str(anchor_node.get("label")) if anchor_node is not None else _NODE_TYPE_LABELS_RU.get(anchor, anchor)
    axis.set_title(f"Происхождение: {anchor_display}", loc="left", fontsize=16, fontweight="bold")
    by_depth: dict[int, list[Mapping[str, Any]]] = {}
    for node in selected:
        by_depth.setdefault(depth.get(str(node.get("node_id")), 0), []).append(node)
    max_depth = max(by_depth) if by_depth else 0
    positions = {}
    for level, level_nodes in by_depth.items():
        x = 0.06 + 0.88 * (max_depth - level) / max(max_depth, 1)  # anchor (depth 0) on the right, sources on the left
        for row, node in enumerate(level_nodes):
            y = 0.5 if len(level_nodes) == 1 else 0.22 + 0.63 * row / max(len(level_nodes) - 1, 1)
            positions[str(node.get("node_id"))] = (x, y)
            node_type = str(node.get("node_type"))
            type_label = _NODE_TYPE_LABELS_RU.get(node_type, node_type)
            # P18 item 10: a directed-ancestry chain is typically short (a
            # handful of nodes), so there is real room to show the full
            # label wrapped across lines instead of truncating it with
            # "[...]" -- only a genuinely long label past 3 wrapped lines
            # still gets an explicit ellipsis (never a silent cut).
            wrapped_lines = wrap(str(node.get("label")), width=24)[:3]
            if len(wrap(str(node.get("label")), width=24)) > 3:
                wrapped_lines[-1] = wrapped_lines[-1].rstrip() + "…"
            body = "\n".join(wrapped_lines)
            axis.text(
                x,
                y,
                f"{type_label}\n{body}",
                ha="center",
                va="center",
                fontsize=8.5,
                bbox={"boxstyle": "round,pad=.35", "facecolor": "#eef6fb" if node.get("node_id") == anchor else "#f6f8f9", "edgecolor": "#355b72"},
            )
        if level in truncated_counts:
            axis.text(x, 0.09, f"(+{truncated_counts[level]} ещё\nна этой глубине)", ha="center", va="center", fontsize=7.5, color="#7b8790")
    for edge in selected_edges:
        source, target = positions.get(str(edge.get("source"))), positions.get(str(edge.get("target")))
        if source and target:
            axis.annotate("", xy=target, xytext=source, arrowprops={"arrowstyle": "->", "color": "#5c7a8a", "lw": 1.0})
    axis.text(
        0.02,
        0.02,
        f"Показан фрагмент из {len(selected)} узлов вокруг «{anchor}» (всего в графе {len(nodes)} узлов; полный граф — result.audit() / inspect(...)).",
        fontsize=9.5,
        color="#58656e",
    )
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


_OBJECT_REPRESENTATION_MAX_CHARS = 900


def _object_representation_text(payload: Mapping[str, Any]) -> Any:
    import matplotlib.pyplot as plt

    text = str(payload.get("raw_excerpt", ""))
    spans = sorted(payload.get("spans", []), key=lambda span: span.get("start", 0))
    truncated = len(text) > _OBJECT_REPRESENTATION_MAX_CHARS
    if truncated:
        text = text[:_OBJECT_REPRESENTATION_MAX_CHARS]
        spans = [span for span in spans if int(span.get("end", 0)) <= len(text)]

    figure, axis = plt.subplots(figsize=(16, 9))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title("Исходный текст с подсветкой измеренных признаков", loc="left", fontsize=20, fontweight="bold")

    if not spans:
        axis.text(0.03, 0.88, "\n".join(wrap(text, 95)[:20]) or "(пустой текст)", fontsize=12, va="top", family="monospace")
        axis.text(0.03, 0.06, "Вклад признаков не удалось сопоставить с исходным текстом.", fontsize=11, color="#7b8790")
        return figure

    char_width, line_height = 0.0105, 0.06
    x, y, cursor = 0.03, 0.90, 0
    segments: list[tuple[str, str | None, str]] = []
    for span in spans:
        start, end = int(span.get("start", 0)), int(span.get("end", 0))
        if start < cursor or start >= len(text):
            continue
        if start > cursor:
            segments.append((text[cursor:start], None, ""))
        direction = str(span.get("direction"))
        color = _COLORS["supported"] if direction == "supports" else _COLORS["conflict"]
        icon = "▲" if direction == "supports" else "▼"
        segments.append((text[start:end], color, icon))
        cursor = end
    segments.append((text[cursor:], None, ""))

    for chunk, mark_color, icon in segments:
        words = chunk.split(" ")
        for index, word in enumerate(words):
            token = word + (" " if index < len(words) - 1 else "")
            if not token:
                continue
            if x + len(token) * char_width > 0.95:
                x, y = 0.03, y - line_height
            style: dict[str, Any] = {"fontsize": 12.5, "va": "top", "family": "monospace"}
            if mark_color:
                style["bbox"] = {"boxstyle": "square,pad=0.15", "facecolor": "#c9ecd8" if icon == "▲" else "#f6cccc", "edgecolor": mark_color, "linewidth": 1.2}
                token = f"{token.strip()}{icon} "
            axis.text(x, y, token, **style)
            x += len(token) * char_width
    footer_y = max(y - 0.10, 0.10)
    axis.text(0.03, footer_y, "▲ поддерживает   ▼ противоречит", fontsize=10, color="#355b72")
    unmapped = payload.get("unmapped_features", [])
    if unmapped:
        axis.text(0.03, max(footer_y - 0.05, 0.02), f"Не найдено в тексте буквально: {', '.join(unmapped)}", fontsize=9.5, color="#7b8790")
    if truncated:
        axis.text(0.03, 0.965, "Показан фрагмент текста (полный текст — в структурированном результате).", fontsize=9.5, color="#7b8790")
    return figure


def _object_representation_tabular(payload: Mapping[str, Any]) -> Any:
    # P18 item 9: this picture is the reader-facing view — capped to the 6
    # strongest supporting + 4 strongest contradicting features, in Russian
    # where a domain_language label was registered. The full, unabridged
    # table is never lost — it stays in the structured spec/audit() output
    # (tabular_rows_original_order), only the rendered image is capped.
    rows = list(payload.get("tabular_rows", []))
    title = "Исходный объект: значения признаков и вклад"
    if not rows:
        return _empty(title, "Признаки объекта недоступны.")
    supports = [item for item in rows if item.get("direction") == "supports"][:6]
    contradicts = [item for item in rows if item.get("direction") == "contradicts"][:4]
    shown = [*supports, *contradicts]
    total_directional = sum(1 for item in rows if item.get("direction") in {"supports", "contradicts"})
    columns = ["Признак", "Значение", "Вклад", "Направление"]
    table_rows = [
        [
            item.get("feature_label") or item.get("feature"),
            item.get("raw_value"),
            "—" if item.get("contribution") is None else f"{item.get('contribution'):+.4f}",
            {"supports": "поддерживает", "contradicts": "противоречит", "unknown": "не измерено"}.get(item.get("direction"), item.get("direction")),
        ]
        for item in shown
    ]
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 6.5))
    axis.axis("off")
    caption_parts = [f"показаны {len(shown)} из {total_directional} направленных признаков — полная таблица в audit()"] if len(shown) < total_directional else []
    limitations = list(payload.get("limitations", []))
    if caption_parts:
        limitations = [*caption_parts, *limitations]
    if limitations:
        axis.text(0.0, 0.98, "; ".join(limitations), fontsize=9, color="#7b8790", va="top")
    table = axis.table(cellText=table_rows, colLabels=columns, loc="upper left", cellLoc="left", colLoc="left", bbox=[0, 0.03, 1, 0.88], colWidths=[0.46, 0.18, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    return figure


def _object_representation_image(payload: Mapping[str, Any]) -> Any:
    import base64
    import io

    import matplotlib.pyplot as plt
    from matplotlib import patches

    title = "Исходное изображение с разметкой evidence"
    encoded = str(payload.get("image_png_base64", ""))
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.set_title(title, loc="left", fontsize=18, fontweight="bold")
    if not encoded or encoded.startswith("["):
        # Redacted (include_raw=False) or never captured — still show
        # dimensions/regions honestly rather than a misleading blank canvas.
        axis.axis("off")
        axis.text(0.02, 0.5, "Изображение недоступно для отображения (include_raw=False или отсутствует).", fontsize=11, color="#7b8790")
    else:
        image_array = plt.imread(io.BytesIO(base64.b64decode(encoded)), format="png")
        axis.imshow(image_array)
        axis.axis("off")
    colors = {"supports": "#2e7d32", "contradicts": "#c62828", "unknown": "#78909c"}
    legend_labels = {"supports": "поддерживает", "contradicts": "противоречит", "unknown": "не измерено"}
    seen_directions: set[str] = set()
    for region in payload.get("image_regions", []):
        row_min, row_max, col_min, col_max = region["bounding_box"]
        direction = str(region.get("direction", "unknown"))
        color = colors.get(direction, "#78909c")
        rectangle = patches.Rectangle(
            (col_min, row_min), col_max - col_min + 1, row_max - row_min + 1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        axis.add_patch(rectangle)
        axis.text(col_min, max(row_min - 3, 0), str(region.get("name", "")), fontsize=9, color=color, fontweight="bold")
        seen_directions.add(direction)
    if seen_directions:
        handles = [patches.Patch(edgecolor=colors[d], facecolor="none", label=legend_labels[d]) for d in sorted(seen_directions)]
        axis.legend(handles=handles, loc="upper right", fontsize=9)
    limitations = payload.get("limitations", [])
    if limitations:
        figure.text(0.02, 0.02, "; ".join(limitations), fontsize=9, color="#7b8790")
    return figure


def _object_representation(spec: Mapping[str, Any]) -> Any:
    payload = spec.get("object_representation")
    if not payload:
        return _empty("Исходный объект с разметкой evidence", "Сырой объект не был передан в explain_one(..., raw_object=...); представление недоступно.")
    modality = str(payload.get("modality", "unknown"))
    if modality == "text":
        return _object_representation_text(payload)
    if modality == "tabular":
        return _object_representation_tabular(payload)
    if modality == "image":
        return _object_representation_image(payload)
    return _empty("Исходный объект с разметкой evidence", f"Визуализация модальности '{modality}' пока не реализована.")


def render_visual_spec(spec: Mapping[str, Any], *, view: str, output_path: str | Path | None = None, selector: str | None = None):
    """Render one focused view from the canonical visual specification.

    ``selector`` only affects the ``provenance`` view — a claim_id
    (``"C-002"``), a node_id (``"action"``), or a node_type (``"claim"``)
    to focus the subgraph on; defaults to ``"action"``.
    """

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
        "provenance": lambda payload: _provenance(payload, selector=selector),
        "audit": _audit,
        "object_representation": _object_representation,
    }
    if view not in renderers:
        raise ValueError(f"unsupported visualization view: {view}")
    return _finish(renderers[view](spec), output_path)
