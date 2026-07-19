from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def render_visual_spec(spec: Mapping[str, Any], *, view: str, output_path: str | Path | None = None):
    """Render an interactive view while preserving claim IDs in hover data."""

    import plotly.graph_objects as go

    aliases = {"dashboard": "explanation_story", "class_atlas": "knowledge_atlas", "counterfactual": "counterfactuals"}
    view = aliases.get(view, view)
    figure = go.Figure()
    if view == "explanation_story":
        stages = list(spec.get("story", []))
        colors = {"supported": "#2f6b4f", "limitation": "#b7791f", "conflict": "#a23b3b", "missing": "#7b8790"}
        for index, stage in enumerate(stages):
            figure.add_trace(
                go.Scatter(
                    x=[index],
                    y=[0],
                    mode="markers+text",
                    text=[stage.get("title")],
                    textposition="top center",
                    marker={"size": 72, "color": colors.get(stage.get("status"), "#7b8790"), "symbol": "square"},
                    customdata=[["<br>".join(stage.get("facts", [])), ", ".join(stage.get("claim_refs", []))]],
                    hovertemplate="%{text}<br>%{customdata[0]}<br>Claims: %{customdata[1]}<extra></extra>",
                    showlegend=False,
                )
            )
            if index:
                figure.add_annotation(x=index - 0.5, y=0, ax=-45, ay=0, text="", showarrow=True, arrowhead=2)
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False, range=[-0.8, 0.8])
        title = "Карта доказательного объяснения"
    elif view == "data_profile":
        items = list(spec.get("data_profile", []))
        figure.add_trace(
            go.Scatter(
                x=[item.get("percentile") for item in items],
                y=[item.get("feature") for item in items],
                mode="markers",
                marker={"size": 14, "color": ["#a23b3b" if item.get("anomaly_status") == "deviation_not_error" else "#2f6b4f" for item in items]},
                customdata=[[item.get("object_value"), item.get("contribution"), item.get("explanation"), ", ".join(item.get("claim_refs", []))] for item in items],
                hovertemplate="%{y}: percentile %{x:.1f}<br>value=%{customdata[0]}<br>contribution=%{customdata[1]}<br>%{customdata[2]}<br>Claims: %{customdata[3]}<extra></extra>",
            )
        )
        figure.update_xaxes(range=[0, 100], title="Перцентиль относительно референса")
        title = "Профиль данных"
    elif view == "training_trace":
        timelines = list(spec.get("training_timeline", []))
        points = timelines[0].get("points", []) if timelines else []
        epochs = [point.get("epoch") for point in points]
        figure.add_trace(go.Scatter(x=epochs, y=[point.get("confidence") for point in points], name="confidence", mode="lines+markers"))
        figure.add_trace(go.Scatter(x=epochs, y=[point.get("loss") for point in points], name="object loss", mode="lines+markers", yaxis="y2"))
        figure.update_layout(yaxis2={"overlaying": "y", "side": "right", "title": "Loss"})
        title = "Траектория обучения: confidence и loss на отдельных шкалах"
    elif view == "decision_evidence":
        decision = dict(spec.get("decision_evidence", {}))
        groups = [
            ("Поддерживает", decision.get("supports", []), "#2f6b4f"),
            ("Противоречит", decision.get("contradicts", []), "#a23b3b"),
            ("Ограничивает", decision.get("limitations", []), "#b7791f"),
        ]
        for index, (name, items, color) in enumerate(groups):
            figure.add_trace(
                go.Scatter(
                    x=[index] * len(items),
                    y=list(range(len(items))),
                    mode="markers",
                    name=name,
                    marker={"size": 18, "color": color},
                    text=[f"{item.get('claim_id')}: {item.get('statement')}" for item in items],
                    hovertemplate="%{text}<extra></extra>",
                )
            )
        title = "Карта evidence для решения"
    elif view in {"similar_cases", "counterfactuals", "rule_ablation", "knowledge_atlas", "provenance", "audit"}:
        payload = {
            "similar_cases": spec.get("similar_cases", []),
            "counterfactuals": spec.get("counterfactuals", []),
            "rule_ablation": [rule for rule in spec.get("knowledge_atlas", {}).get("rules", []) if rule.get("counterfactual_effect")],
            "knowledge_atlas": spec.get("knowledge_atlas", {}),
            "provenance": {"nodes": spec.get("provenance_nodes", []), "edges": spec.get("provenance_edges", [])},
            "audit": spec.get("overview", {}),
        }[view]
        figure.add_annotation(text=f"{view}<br><br>Подробные записи: {len(payload) if isinstance(payload, list) else len(payload.keys())}", x=0.5, y=0.5, showarrow=False, font={"size": 18})
        title = view
    else:
        raise ValueError(f"unsupported visualization view: {view}")
    figure.update_layout(title=title, template="plotly_white", width=1600, height=900, font={"family": "IBM Plex Sans, sans-serif"})
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
