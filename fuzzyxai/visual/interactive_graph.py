from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Tuple

from fuzzyxai.core.explanation_object import ExplanationObject
from fuzzyxai.visual.composition_graph import edge_report


def _edge_color(gamma: float) -> str:
    if gamma < 0.20:
        return '#2ca02c'
    if gamma < 0.45:
        return '#ff7f0e'
    return '#d62728'


def composition_plotly_figure(edges: Sequence[Tuple[str, ExplanationObject, str, ExplanationObject]], beta: Mapping[str, float]):
    """Build a Plotly figure for an explanation-composition graph.

    The implementation uses a deterministic left-to-right layout to avoid extra
    graph dependencies.
    """
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise RuntimeError('plotly is required for interactive graph rendering') from exc

    rows = edge_report(edges, beta)
    nodes = []
    for source, _, target, _ in edges:
        if source not in nodes:
            nodes.append(source)
        if target not in nodes:
            nodes.append(target)
    pos = {name: (idx, 0) for idx, name in enumerate(nodes)}

    fig = go.Figure()
    for row in rows:
        x0, y0 = pos[row['source']]
        x1, y1 = pos[row['target']]
        gamma = float(row['gamma'])
        text = (
            f"{row['source']} → {row['target']}<br>"
            f"gamma={gamma:.4f}<br>severity={row['severity']}<br>"
            f"left rules={row['left_active_rules']}<br>right rules={row['right_active_rules']}<br>"
            f"Delta=({row['left_delta']:.4f},{row['right_delta']:.4f})"
        )
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode='lines',
            line=dict(width=2 + 6 * gamma, color=_edge_color(gamma)),
            hoverinfo='text', text=[text, text], showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
        mode='markers+text', text=nodes, textposition='bottom center',
        marker=dict(size=28, color='#1f77b4'), hoverinfo='text',
        showlegend=False,
    ))
    fig.update_layout(
        title='FuzzyXAI composition graph: edge color encodes semantic disagreement γ',
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=420, margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='white',
    )
    return fig


def save_composition_html(edges, beta, path: str | Path) -> None:
    fig = composition_plotly_figure(edges, beta)
    fig.write_html(str(path), include_plotlyjs='cdn')
