from __future__ import annotations

from pathlib import Path

from .utils import add_footer, apply_visual_style, ensure_parent, footer_text, read_csv, write_html_with_image


def render_operator_trace_heatmap(
    results_csv: str | Path,
    out: str | Path,
    html_out: str | Path | None = None,
    *,
    compact: bool = False,
) -> Path:
    rows = read_csv(results_csv)
    out = ensure_parent(out)
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib and numpy are required") from exc
    cols = ["gamma", "delta", "rho", "uncertainty_component", "quality_component", "reduction_component", "conflict_component"]
    if compact:
        cols = ["gamma", "delta", "rho", "uncertainty_component", "reduction_component", "quality_component"]
    matrix = np.array([[float(row.get(col, 0.0) or 0.0) for col in cols] for row in rows])
    fig, ax = plt.subplots(figsize=(13.5, 8.2 if compact else 9.2))
    apply_visual_style(fig, ax)
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(rows)))
    labels = [row["experiment_id"] for row in rows]
    ax.set_yticklabels([label[:32] for label in labels], fontsize=8 if compact else 7)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right")
    if compact:
        for i, row in enumerate(rows):
            ax.text(len(cols) + 0.15, i, row.get("action_id", ""), va="center", fontsize=7, color="#37474f")
        ax.set_xlim(-0.5, len(cols) + 1.9)
        ax.text(len(cols) + 0.15, -0.85, "action", fontsize=8, weight="bold")
    ax.set_title("Operator Trace Heatmap Compact" if compact else "Operator Trace Heatmap Full", weight="bold")
    fig.colorbar(im, ax=ax, label="risk / component value")
    add_footer(fig, footer_text(source_commit=rows[0].get("source_commit") if rows else None, verifier="passed"))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    if html_out:
        write_html_with_image(out, html_out, title="Operator Trace Heatmap", summary={"experiments": len(rows)})
    return out
