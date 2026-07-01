from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import as_float, ensure_parent, read_json, status_color, write_html_with_image


def render_route_sankey(route: str | Path | dict[str, Any], out: str | Path, html_out: str | Path | None = None) -> Path:
    data = read_json(route) if not isinstance(route, dict) else route
    out = ensure_parent(out)
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required") from exc

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    computed = data.get("computed_result", {})
    action = computed.get("action_id") or data.get("final_action_id") or data.get("final_action", "")
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.axis("off")
    ax.set_xlim(0, max(len(nodes), 1))
    ax.set_ylim(0, 1)
    fig.patch.set_facecolor("#fbfbf8")
    ax.text(0, 1.06, "FuzzyXAI Operator Route Sankey", fontsize=19, weight="bold", color="#14213d")
    ax.text(
        0,
        1.0,
        f"scenario={data.get('scenario_id')}  action={action}  gamma={computed.get('gamma')}  delta={computed.get('delta')}  rho={computed.get('rho')}",
        fontsize=10,
        color="#41505f",
    )
    positions: dict[str, tuple[float, float]] = {}
    for idx, node in enumerate(nodes):
        x = idx + 0.12
        y = 0.45
        positions[node["node_id"]] = (x + 0.35, y + 0.13)
        output = node.get("output_values") or {}
        value = node.get("value") or ", ".join(f"{k}={v}" for k, v in list(output.items())[:2])
        fill = "#fff6db" if node.get("status") in {"warning", "audit"} else "#e8f3ff"
        edge_color = status_color(str(action)) if node["node_id"] in {"risk", "action"} else "#506070"
        box = FancyBboxPatch((x, y), 0.7, 0.26, boxstyle="round,pad=0.02", facecolor=fill, edgecolor=edge_color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + 0.04, y + 0.21, node.get("title_ru") or node.get("title") or node["node_id"], fontsize=8.5, weight="bold", va="top")
        ax.text(x + 0.04, y + 0.10, str(value)[:54], fontsize=7.2, color="#263238", va="top")
    for edge in edges:
        src = positions.get(edge["source_node_id"])
        dst = positions.get(edge["target_node_id"])
        if not src or not dst:
            continue
        values = edge.get("passed_values") or {}
        numeric = [as_float(v) for v in values.values() if isinstance(v, (int, float)) or str(v).replace(".", "", 1).isdigit()]
        width = 1.0 + 5.0 * max(numeric) if numeric else 1.2
        arrow = FancyArrowPatch(src, dst, arrowstyle="-|>", mutation_scale=14, linewidth=min(width, 5), color="#55718a", alpha=0.75)
        ax.add_patch(arrow)
        label = "; ".join(f"{k}={v}" for k, v in list(values.items())[:2])
        if label:
            ax.text((src[0] + dst[0]) / 2, 0.35, label[:42], fontsize=7, ha="center", color="#37474f", rotation=18)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    if html_out:
        write_html_with_image(out, html_out, title="Operator Route Sankey", summary=computed)
    return out
