from __future__ import annotations

from pathlib import Path

from .utils import ensure_parent, read_csv, status_color, write_html_with_image


def render_gamma_delta_action_map(results_csv: str | Path, out: str | Path, html_out: str | Path | None = None) -> Path:
    rows = read_csv(results_csv)
    out = ensure_parent(out)
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required") from exc

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#fbfbf8")
    zones = [
        (0, 0, 0.35, 0.35, "accept", "#e6f4ea"),
        (0, 0.35, 0.60, 0.25, "lower_confidence", "#fff4d6"),
        (0.35, 0, 0.25, 0.35, "lower_confidence", "#fff4d6"),
        (0, 0.60, 1, 0.40, "audit", "#ffe4d6"),
        (0.60, 0, 0.40, 0.60, "audit", "#ffe4d6"),
    ]
    for x, y, w, h, label, color in zones:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="white", linewidth=1.2, zorder=0))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9, color="#51606f", alpha=0.75)
    for row in rows:
        gamma = float(row["gamma"])
        delta = float(row["delta"])
        action = row.get("action_id", "")
        ax.scatter(gamma, delta, s=68, color=status_color(action), edgecolor="#1c252e", linewidth=0.5, zorder=3)
        ax.text(gamma + 0.012, delta + 0.012, row.get("experiment_id", "")[:22], fontsize=6.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("gamma: disagreement / uncertainty")
    ax.set_ylabel("delta: explanation reduction loss")
    ax.set_title("FuzzyXAI Gamma-Delta Action Map", weight="bold")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    if html_out:
        write_html_with_image(out, html_out, title="Gamma-Delta Action Map", summary={"experiments": len(rows)})
    return out
