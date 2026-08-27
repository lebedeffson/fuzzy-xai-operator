from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

_EFFECT_COLORS = {
    "favorable": "#2f6b4f",
    "adverse": "#a23b3b",
    "neutral": "#355b72",
    "mixed": "#b7791f",
    "unknown": "#7b8790",
}


def _finish(figure: Any, view: str, output_path: str | Path | None):
    figure.update_layout(template="plotly_white", width=1600, height=900, font={"family": "IBM Plex Sans, sans-serif"})
    if output_path is None:
        return figure
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".html":
        figure.write_html(output, include_plotlyjs=True, div_id=f"fuzzyxai-{view.replace('_', '-')}")
    else:
        try:
            figure.write_image(output, width=1600, height=900)
        except ValueError as exc:
            raise RuntimeError("Plotly raster export requires the optional kaleido package; use an .html output instead") from exc
    return output


def _story(go: Any, spec: Mapping[str, Any]):
    figure = go.Figure()
    stages = list(spec.get("story", []))
    for index, stage in enumerate(stages):
        status = stage.get("evidence_status", "insufficient_evidence")
        effect = stage.get("effect", "unknown")
        figure.add_trace(go.Scatter(
            x=[index], y=[0], mode="markers+text", text=[stage.get("title")], textposition="top center",
            marker={"size": 76, "color": _EFFECT_COLORS.get(effect, "#7b8790"), "symbol": "square", "line": {"width": 4 if status != "supported" else 1, "color": "#b7791f" if status != "supported" else "#ffffff"}},
            customdata=[[status, effect, stage.get("severity"), "<br>".join(stage.get("facts", [])), ", ".join(stage.get("claim_refs", []))]],
            hovertemplate="%{text}<br>Evidence: %{customdata[0]}<br>Effect: %{customdata[1]}<br>Severity: %{customdata[2]}<br>%{customdata[3]}<br>Claims: %{customdata[4]}<extra></extra>", showlegend=False,
        ))
        if index:
            figure.add_shape(type="line", x0=index - 0.82, x1=index - 0.18, y0=0, y1=0, line={"color": "#355b72", "width": 2})
            figure.add_annotation(x=index - 0.18, y=0, ax=index - 0.3, ay=0, xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True, arrowhead=2)
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False, range=[-0.8, 0.8])
    figure.update_layout(title="Карта доказательного объяснения")
    return figure


def _data_profile(go: Any, spec: Mapping[str, Any]):
    items = list(spec.get("data_profile", []))
    figure = go.Figure()
    for item in items:
        figure.add_trace(go.Scatter(
            x=[5, 95], y=[item.get("feature")] * 2, mode="lines", line={"width": 14, "color": "#d2dce1"}, showlegend=False, hoverinfo="skip",
        ))
    figure.add_trace(go.Scatter(
        x=[item.get("percentile") for item in items], y=[item.get("feature") for item in items], mode="markers", name="Объект",
        marker={"size": 16, "color": ["#a23b3b" if item.get("anomaly_status") == "deviation_not_error" else "#2f6b4f" for item in items]},
        customdata=[[item.get("object_value"), item.get("contribution"), item.get("subgroup_interval"), item.get("explanation"), ", ".join(item.get("claim_refs", []))] for item in items],
        hovertemplate="%{y}: percentile %{x:.1f}<br>value=%{customdata[0]}<br>contribution=%{customdata[1]}<br>subgroup=%{customdata[2]}<br>%{customdata[3]}<br>Claims: %{customdata[4]}<extra></extra>",
    ))
    figure.update_xaxes(range=[0, 100], title="Перцентиль относительно референса")
    figure.update_layout(title="Профиль данных: референс, подгруппа и объясняемый объект")
    return figure


def _training_trace(go: Any, spec: Mapping[str, Any]):
    from plotly.subplots import make_subplots

    timelines = list(spec.get("training_timeline", []))
    points = timelines[0].get("points", []) if timelines else []
    epochs = [point.get("epoch") for point in points]
    figure = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.025, subplot_titles=("Correctness", "Confidence", "Object loss", "Margin", "Prototype / subgroup / global", "Rule activations"))
    figure.add_trace(go.Scatter(x=epochs, y=[1 if point.get("correct") else 0 for point in points], mode="markers", marker={"color": ["#2f6b4f" if point.get("correct") else "#a23b3b" for point in points]}, name="correct"), row=1, col=1)
    for row, key, color in ((2, "confidence", "#286f91"), (3, "loss", "#b7791f"), (4, "margin", "#70558c")):
        figure.add_trace(go.Scatter(x=epochs, y=[point.get(key) for point in points], mode="lines+markers", name=key, line={"color": color}), row=row, col=1)
    for key, name, color in (("prototype_distance", "prototype distance", "#5a7582"), ("subgroup_metric", "subgroup metric", "#a23b3b"), ("global_metric", "global metric", "#2f6b4f")):
        figure.add_trace(go.Scatter(x=epochs, y=[point.get(key) for point in points], mode="lines+markers", name=name, line={"color": color}), row=5, col=1)
    rule_ids = sorted({item.get("rule_id") for point in points for item in point.get("rule_activations", []) if item.get("rule_id")})
    for rule_id in rule_ids:
        values = [next((item.get("activation") for item in point.get("rule_activations", []) if item.get("rule_id") == rule_id), None) for point in points]
        figure.add_trace(go.Scatter(x=epochs, y=values, mode="lines+markers", name=str(rule_id)), row=6, col=1)
    for point in points:
        if point.get("forgetting"):
            figure.add_vline(x=point.get("epoch"), line_dash="dash", line_color="#a23b3b")
    figure.update_layout(title=f"Траектория обучения объекта {timelines[0].get('object_id') if timelines else 'недоступна'}")
    return figure


def _decision(go: Any, spec: Mapping[str, Any]):
    decision = dict(spec.get("decision_evidence", {}))
    figure = go.Figure()
    for index, (name, key, color) in enumerate((("Поддерживает", "supports", "#2f6b4f"), ("Противоречит", "contradicts", "#a23b3b"), ("Ограничивает", "limitations", "#b7791f"))):
        items = decision.get(key, [])
        figure.add_trace(go.Scatter(x=[index] * len(items), y=list(range(len(items))), mode="markers+text", name=name, marker={"size": 18, "color": color}, text=[f"{item.get('claim_id')}: {item.get('statement')}" for item in items], textposition="middle right", customdata=[[item.get("evidence_status"), item.get("effect"), item.get("severity"), ", ".join(item.get("evidence_refs", []))] for item in items], hovertemplate="%{text}<br>Evidence=%{customdata[0]}, effect=%{customdata[1]}, severity=%{customdata[2]}<br>Refs: %{customdata[3]}<extra></extra>"))
    figure.update_xaxes(tickvals=[0, 1, 2], ticktext=["Поддерживает", "Противоречит", "Ограничивает"], range=[-0.4, 3])
    figure.update_layout(title="Карта evidence для решения")
    return figure


def _knowledge_atlas(go: Any, spec: Mapping[str, Any]):
    atlas = dict(spec.get("knowledge_atlas", {}))
    concepts, rules = list(atlas.get("concepts", [])), list(atlas.get("rules", []))
    labels = [f"Класс: {item.get('class_name')}" for item in concepts] + [f"Правило: {item.get('rule_id')}" for item in rules]
    concept_index = {item.get("class_id"): index for index, item in enumerate(concepts)}
    rule_index = {item.get("rule_id"): len(concepts) + index for index, item in enumerate(rules)}
    sources, targets, values = [], [], []
    for concept in concepts:
        for rule_id in concept.get("primary_rules", []):
            if rule_id in rule_index:
                sources.append(rule_index[rule_id])
                targets.append(concept_index[concept.get("class_id")])
                values.append(1)
    if not labels:
        labels = ["Evidence отсутствует"]
    figure = go.Figure(go.Sankey(node={"label": labels, "color": ["#355b72"] * len(concepts) + ["#87a8b8"] * len(rules)}, link={"source": sources, "target": targets, "value": values, "customdata": ["rule supports class concept"] * len(values), "hovertemplate": "%{customdata}<extra></extra>"}))
    figure.update_layout(title=f"Атлас знаний: {atlas.get('source_rule_count', 0)} правил, показано {atlas.get('displayed_rule_count', 0)}")
    return figure


def _similar_cases(go: Any, spec: Mapping[str, Any]):
    items = list(spec.get("similar_cases", []))
    figure = go.Figure(go.Bar(
        x=[item.get("score") for item in items], y=[item.get("reference_object_id") for item in items], orientation="h",
        marker={"color": ["#a23b3b" if item.get("is_counterexample") else "#2f6b4f" for item in items]},
        customdata=[[item.get("method"), item.get("representation"), ", ".join(item.get("matched_features", [])), ", ".join(item.get("different_features", [])), "; ".join(item.get("limitations", [])), "counterexample" if item.get("is_counterexample") else "supporting case", "; ".join(f"{pair[0]}={pair[1]}" for pair in item.get("media_artifacts", []))] for item in items],
        hovertemplate="Object %{y}<br>score=%{x:.3f}<br>%{customdata[5]}<br>method=%{customdata[0]}<br>representation=%{customdata[1]}<br>matched=%{customdata[2]}<br>different=%{customdata[3]}<br>media=%{customdata[6]}<br>limits=%{customdata[4]}<extra></extra>",
    ))
    figure.update_xaxes(range=[0, 1], title="Similarity in the declared representation (not probability)")
    figure.update_layout(title="Похожие случаи и контрпримеры")
    return figure


def _counterfactuals(go: Any, spec: Mapping[str, Any]):
    items = list(spec.get("counterfactuals", []))
    labels, effects, custom = [], [], []
    for cf_index, item in enumerate(items):
        for change in item.get("changed_features", []):
            labels.append(f"CF{cf_index + 1}: {change.get('feature')}")
            effects.append(item.get("observed_effect") if item.get("observed_effect") is not None else 0)
            custom.append([change.get("source_value"), change.get("target_value"), item.get("target_prediction"), item.get("minimality"), item.get("plausibility"), item.get("actionability")])
    figure = go.Figure(go.Bar(x=effects, y=labels, orientation="h", marker={"color": "#355b72"}, customdata=custom, hovertemplate="%{y}<br>%{customdata[0]} → %{customdata[1]}<br>target=%{customdata[2]}<br>minimality=%{customdata[3]}, plausibility=%{customdata[4]}<br>%{customdata[5]}<extra></extra>"))
    figure.update_layout(title="Контрфактические изменения и наблюдаемый эффект")
    return figure


def _rule_ablation(go: Any, spec: Mapping[str, Any]):
    items = list(spec.get("rule_ablations", []))
    labels = [f"{item.get('rule_id')} · {item.get('metric')}" for item in items]
    figure = go.Figure()
    figure.add_trace(go.Bar(name="С правилом", x=labels, y=[item.get("with_rule") for item in items], marker_color="#2f6b4f", customdata=[[item.get("scope"), item.get("difference")] for item in items], hovertemplate="%{x}<br>with=%{y:.4f}<br>scope=%{customdata[0]}<br>difference=%{customdata[1]:+.4f}<extra></extra>"))
    figure.add_trace(go.Bar(name="Без правила", x=labels, y=[item.get("without_rule") for item in items], marker_color="#a23b3b"))
    figure.update_layout(barmode="group", title="Rule ablation: train / validation / test / subgroup")
    return figure


def _provenance(go: Any, spec: Mapping[str, Any]):
    nodes, edges = list(spec.get("provenance_nodes", [])), list(spec.get("provenance_edges", []))
    index = {node.get("node_id"): position for position, node in enumerate(nodes)}
    valid = [edge for edge in edges if edge.get("source") in index and edge.get("target") in index]
    colors = {"data": "#5a8f73", "claim": "#355b72", "diagnostic": "#b7791f", "action": "#a23b3b"}
    figure = go.Figure(go.Sankey(node={"label": [f"{node.get('node_type')}: {node.get('label')}" for node in nodes], "color": [colors.get(node.get("node_type"), "#a8b8c0") for node in nodes]}, link={"source": [index[edge.get("source")] for edge in valid], "target": [index[edge.get("target")] for edge in valid], "value": [1] * len(valid), "customdata": [edge.get("relation") for edge in valid], "hovertemplate": "%{customdata}<extra></extra>"}))
    figure.update_layout(title="Трассируемость: evidence → claim → diagnostic → action")
    return figure


def _audit(go: Any, spec: Mapping[str, Any]):
    overview, audit = dict(spec.get("overview", {})), dict(spec.get("audit", {}))
    counts = dict(overview.get("claim_counts", {}))
    headers = ["Проверка", "Значение"]
    rows = [
        ["Schema", spec.get("schema_version")], ["Graph validation", "PASS" if audit.get("graph_valid") else "FAIL"],
        ["Action", overview.get("action")], ["Explanation level", overview.get("explanation_level", {}).get("level")],
        ["Claims", counts.get("total")], ["Supported", counts.get("supported")], ["Contested", counts.get("contested")],
        ["Insufficient evidence", counts.get("insufficient_evidence")], ["Graph nodes", audit.get("node_count")], ["Graph edges", audit.get("edge_count")],
    ]
    figure = go.Figure(go.Table(header={"values": headers, "fill_color": "#dce7ec", "align": "left"}, cells={"values": [[row[0] for row in rows], [row[1] for row in rows]], "align": "left", "height": 34}))
    figure.update_layout(title="Аудит объяснения")
    return figure


_OBJECT_REPRESENTATION_MAX_CHARS = 900


def _object_representation_text(go: Any, payload: Mapping[str, Any]) -> Any:
    from html import escape as _escape

    text = str(payload.get("raw_excerpt", ""))
    spans = sorted(payload.get("spans", []), key=lambda span: span.get("start", 0))
    truncated = len(text) > _OBJECT_REPRESENTATION_MAX_CHARS
    if truncated:
        text = text[:_OBJECT_REPRESENTATION_MAX_CHARS]
        spans = [span for span in spans if int(span.get("end", 0)) <= len(text)]

    pieces: list[str] = []
    cursor = 0
    for span in spans:
        start, end = int(span.get("start", 0)), int(span.get("end", 0))
        if start < cursor:
            continue
        pieces.append(_escape(text[cursor:start]))
        supports = span.get("direction") == "supports"
        color = _EFFECT_COLORS["favorable"] if supports else _EFFECT_COLORS["adverse"]
        icon = "▲" if supports else "▼"
        pieces.append(f'<span style="background-color:{"#c9ecd8" if supports else "#f6cccc"};color:{color}"><b>{_escape(text[start:end])}{icon}</b></span>')
        cursor = end
    pieces.append(_escape(text[cursor:]))
    body = "".join(pieces).replace("\n", "<br>")
    if not spans:
        body = _escape(text).replace("\n", "<br>") or "(пустой текст)"

    figure = go.Figure()
    figure.add_annotation(
        x=0, y=1, xref="paper", yref="paper", showarrow=False, align="left", xanchor="left", yanchor="top",
        text=body, font={"size": 15}, bordercolor="#c8d3d8", borderwidth=1, borderpad=10, bgcolor="#fbfcfc",
    )
    unmapped = payload.get("unmapped_features", [])
    legend = "<b>▲ поддерживает</b> &nbsp; <b>▼ противоречит</b>"
    if unmapped:
        legend += f" &nbsp;·&nbsp; не найдено буквально: {_escape(', '.join(unmapped))}"
    if not spans:
        legend = "Вклад признаков не удалось сопоставить с исходным текстом."
    figure.add_annotation(x=0, y=0, xref="paper", yref="paper", showarrow=False, align="left", xanchor="left", yanchor="bottom", text=legend, font={"size": 11, "color": "#58656e"})
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    title = f"Исходный текст объекта {payload.get('object_id', '')} с evidence-разметкой"
    if truncated:
        title += " (показан фрагмент)"
    figure.update_layout(title=title)
    return figure


def _object_representation_tabular(go: Any, payload: Mapping[str, Any]) -> Any:
    rows = list(payload.get("tabular_rows", []))[:25]
    if not rows:
        figure = go.Figure()
        figure.update_layout(title="Признаки объекта недоступны.")
        return figure
    labels = {"supports": "поддерживает", "contradicts": "противоречит", "unknown": "не измерено"}
    header = ["Признак", "Значение", "Вклад", "Направление"]
    cells = [
        [item.get("feature") for item in rows],
        [item.get("raw_value") for item in rows],
        ["—" if item.get("contribution") is None else f"{item.get('contribution'):+.4f}" for item in rows],
        [labels.get(item.get("direction"), item.get("direction")) for item in rows],
    ]
    figure = go.Figure(go.Table(header={"values": header, "fill_color": "#dce7ec", "align": "left"}, cells={"values": cells, "align": "left", "height": 34}))
    limitations = payload.get("limitations", [])
    subtitle = f" · {'; '.join(limitations)}" if limitations else ""
    figure.update_layout(title=f"Исходный объект {payload.get('object_id', '')}: значения признаков и вклад{subtitle}")
    return figure


def _object_representation_image(go: Any, payload: Mapping[str, Any]) -> Any:
    encoded = str(payload.get("image_png_base64", ""))
    figure = go.Figure()
    labels = {"supports": "поддерживает", "contradicts": "противоречит", "unknown": "не измерено"}
    colors = {"supports": "#2e7d32", "contradicts": "#c62828", "unknown": "#78909c"}
    if not encoded or encoded.startswith("["):
        figure.update_layout(title="Изображение недоступно для отображения (include_raw=False или отсутствует).")
        return figure
    width, height = payload.get("image_width"), payload.get("image_height")
    figure.add_layout_image(
        source=f"data:image/png;base64,{encoded}",
        xref="x", yref="y", x=0, y=0, sizex=width, sizey=height, sizing="stretch", layer="below",
    )
    seen_directions: set[str] = set()
    for region in payload.get("image_regions", []):
        row_min, row_max, col_min, col_max = region["bounding_box"]
        direction = str(region.get("direction", "unknown"))
        color = colors.get(direction, "#78909c")
        figure.add_shape(type="rect", x0=col_min, x1=col_max + 1, y0=row_min, y1=row_max + 1, line={"color": color, "width": 2})
        figure.add_annotation(x=col_min, y=max(row_min - 3, 0), text=str(region.get("name", "")), showarrow=False, font={"color": color, "size": 10})
        seen_directions.add(direction)
    for direction in sorted(seen_directions):
        figure.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker={"color": colors[direction]}, name=labels[direction]))
    figure.update_xaxes(visible=False, range=[0, width])
    figure.update_yaxes(visible=False, range=[height, 0])
    limitations = payload.get("limitations", [])
    subtitle = f" · {'; '.join(limitations)}" if limitations else ""
    figure.update_layout(title=f"Исходное изображение {payload.get('object_id', '')} с разметкой evidence{subtitle}")
    return figure


def _object_representation(go: Any, spec: Mapping[str, Any]) -> Any:
    payload = dict(spec.get("object_representation") or {})
    if not payload:
        figure = go.Figure()
        figure.update_layout(title="Сырой объект не был передан в explain_one(..., raw_object=...); представление недоступно.")
        return figure
    modality = str(payload.get("modality", "unknown"))
    if modality == "text":
        return _object_representation_text(go, payload)
    if modality == "tabular":
        return _object_representation_tabular(go, payload)
    if modality == "image":
        return _object_representation_image(go, payload)
    figure = go.Figure()
    figure.update_layout(title=f"Визуализация модальности '{modality}' пока не реализована.")
    return figure


def render_visual_spec(spec: Mapping[str, Any], *, view: str, output_path: str | Path | None = None, selector: str | None = None):
    """Render one real interactive view from the canonical VisualSpec.

    ``selector`` is accepted for interface parity with the matplotlib
    backend's focused ``provenance`` view; the plotly ``provenance``
    renderer does not yet use it (still renders the sampled full-graph
    view) — a known follow-up, not implemented in this pass.
    """

    import plotly.graph_objects as go

    aliases = {"dashboard": "explanation_story", "class_atlas": "knowledge_atlas", "counterfactual": "counterfactuals"}
    selected = aliases.get(view, view)
    renderers = {
        "explanation_story": _story,
        "data_profile": _data_profile,
        "training_trace": _training_trace,
        "knowledge_atlas": _knowledge_atlas,
        "decision_evidence": _decision,
        "similar_cases": _similar_cases,
        "counterfactuals": _counterfactuals,
        "rule_ablation": _rule_ablation,
        "provenance": _provenance,
        "audit": _audit,
        "object_representation": _object_representation,
    }
    if selected not in renderers:
        raise ValueError(f"unsupported visualization view: {selected}")
    return _finish(renderers[selected](go, spec), selected, output_path)
