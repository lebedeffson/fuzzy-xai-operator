from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import ensure_parent, read_json, status_color, write_html_with_image


def render_action_boundary(route: str | Path | dict[str, Any], out: str | Path, html_out: str | Path | None = None) -> Path:
    data = read_json(route) if not isinstance(route, dict) else route
    computed = data.get("computed_result", {})
    rho = float(computed.get("rho", 0.0))
    action = computed.get("action_id") or data.get("final_action_id") or data.get("final_action", "")
    out = ensure_parent(out)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required") from exc
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    bands = [(0, 0.35, "accept", "#e6f4ea"), (0.35, 0.60, "lower_confidence", "#fff4d6"), (0.60, 1.0, "audit/block", "#ffe4d6")]
    for start, end, label, color in bands:
        ax.barh(0.5, end - start, left=start, height=0.32, color=color, edgecolor="white")
        ax.text((start + end) / 2, 0.5, label, ha="center", va="center", fontsize=9)
    ax.axvline(rho, color=status_color(str(action)), linewidth=3)
    ax.text(rho, 0.78, f"rho={rho:.3f}\naction={action}", ha="center", fontsize=9, color="#1c252e")
    ax.text(0.35, 0.18, f"distance to accept: {max(rho - 0.35, 0):.3f}", ha="center", fontsize=8)
    ax.text(0.60, 0.18, f"distance to audit: {max(0.60 - rho, 0):.3f}", ha="center", fontsize=8)
    ax.set_title("Action Boundary Plot", weight="bold")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    if html_out:
        write_html_with_image(out, html_out, title="Action Boundary Plot", summary={"rho": rho, "action": action})
    return out
