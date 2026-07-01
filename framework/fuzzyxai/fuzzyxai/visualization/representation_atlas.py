from __future__ import annotations

from pathlib import Path

from .utils import ensure_parent, read_csv, write_html_with_image


def render_representation_atlas(results_csv: str | Path, out: str | Path, html_out: str | Path | None = None) -> Path:
    rows = read_csv(results_csv)
    out = ensure_parent(out)
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib and numpy are required") from exc
    tasks = sorted({row["task_type"] for row in rows})
    perturbations = sorted({row["perturbation"] for row in rows})
    classes = {"F0": 0, "F_int": 1, "NAS": 2, "F_ML": 3}
    matrix = np.full((len(tasks), len(perturbations)), -1.0)
    labels = [["" for _ in perturbations] for _ in tasks]
    for row in rows:
        i = tasks.index(row["task_type"])
        j = perturbations.index(row["perturbation"])
        matrix[i, j] = classes.get(row["representation_class"], -1)
        labels[i][j] = row["representation_class"]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=3, aspect="auto")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_xticks(range(len(perturbations)))
    ax.set_xticklabels(perturbations, rotation=35, ha="right")
    for i in range(len(tasks)):
        for j in range(len(perturbations)):
            if labels[i][j]:
                ax.text(j, i, labels[i][j], ha="center", va="center", color="white", fontsize=8, weight="bold")
    ax.set_title("Representation Class Atlas", weight="bold")
    fig.colorbar(im, ax=ax, ticks=list(classes.values()), label="F0 / F_int / NAS / F_ML")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    if html_out:
        write_html_with_image(out, html_out, title="Representation Class Atlas", summary={"task_types": len(tasks), "perturbations": len(perturbations)})
    return out
