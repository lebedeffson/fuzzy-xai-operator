from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import ensure_parent, read_json, status_color, write_html_with_image


def render_risk_waterfall(trace: str | Path | dict[str, Any], out: str | Path, html_out: str | Path | None = None) -> Path:
    data = read_json(trace) if not isinstance(trace, dict) else trace
    computed = data.get("computed_result", {})
    out = ensure_parent(out)
    components = [
        ("uncertainty", float(computed.get("uncertainty_component", computed.get("uncertainty", 0.0)))),
        ("reduction", float(computed.get("reduction_component", computed.get("delta", 0.0)))),
        ("quality", float(computed.get("quality_component", computed.get("quality_penalty", 0.0)))),
        ("conflict", float(computed.get("conflict_component", 0.0))),
        ("interval", float(computed.get("interval_component", 0.0))),
    ]
    rho = float(computed.get("rho", max(v for _, v in components)))
    action = computed.get("action_id", "")
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required") from exc
    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [k for k, _ in components] + ["rho"]
    values = [v for _, v in components] + [rho]
    colors = ["#7aa6c2", "#f2a900", "#b887ff", "#d75a00", "#607d8b", status_color(str(action))]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("risk component value")
    ax.set_title(f"Risk Waterfall: rho={rho:.3f}, action={action}", weight="bold")
    for i, value in enumerate(values):
        ax.text(i, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)
    ax.text(0.02, 0.93, f"dominant={computed.get('risk_dominant_component')}", transform=ax.transAxes, color="#37474f")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    if html_out:
        write_html_with_image(out, html_out, title="Risk Waterfall", summary=computed)
    return out
