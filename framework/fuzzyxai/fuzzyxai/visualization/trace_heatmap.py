from __future__ import annotations

from pathlib import Path

from .utils import ensure_parent, read_csv, write_html_with_image


def render_operator_trace_heatmap(results_csv: str | Path, out: str | Path, html_out: str | Path | None = None) -> Path:
    rows = read_csv(results_csv)
    out = ensure_parent(out)
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib and numpy are required") from exc
    cols = ["gamma", "delta", "rho", "uncertainty_component", "quality_component", "reduction_component", "conflict_component"]
    matrix = np.array([[float(row.get(col, 0.0) or 0.0) for col in cols] for row in rows])
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row["experiment_id"] for row in rows], fontsize=7)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_title("Operator Trace Heatmap", weight="bold")
    fig.colorbar(im, ax=ax, label="risk / component value")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    if html_out:
        write_html_with_image(out, html_out, title="Operator Trace Heatmap", summary={"experiments": len(rows)})
    return out
