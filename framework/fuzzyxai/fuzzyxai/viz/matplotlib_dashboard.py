from __future__ import annotations

from pathlib import Path

from .operator_state import OperatorRouteState


COLORS = {
    "passed": ("#e9f7ef", "#16803a"),
    "warning": ("#fff4dc", "#b26a00"),
    "blocked": ("#fde8e8", "#b91c1c"),
    "info": ("#eaf2ff", "#2563eb"),
}


def _wrap(text: str, width: int = 34) -> str:
    words = str(text).split()
    lines: list[str] = []
    line: list[str] = []
    for word in words:
        if sum(len(w) for w in line) + len(line) + len(word) > width:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def render_operator_dashboard(route: OperatorRouteState, output_path: str | Path) -> Path:
    """Render a static operator-route dashboard for dissertation figures."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for operator dashboard rendering") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = route.nodes
    cols = 4
    rows = (len(nodes) + cols - 1) // cols
    fig_w = 18
    fig_h = 4.2 * rows + 1.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows + 0.8)
    ax.axis("off")
    fig.patch.set_facecolor("#f7f7f4")

    ax.text(
        0,
        rows + 0.55,
        route.title,
        fontsize=20,
        weight="bold",
        color="#14213d",
        va="top",
    )
    ax.text(
        0,
        rows + 0.22,
        f"source_commit: {route.source_commit}    verifier: {route.verifier_status}    action: {route.final_action}",
        fontsize=11,
        color="#46515c",
        va="top",
    )

    boxes: list[dict[str, float]] = []
    for idx, node in enumerate(nodes):
        row = idx // cols
        col_in_row = idx % cols
        col = col_in_row if row % 2 == 0 else cols - 1 - col_in_row
        x = col + 0.08
        y = rows - row - 0.78
        boxes.append({"x": x, "y": y, "w": 0.84, "h": 0.62, "row": row, "col": col})
        fill, edge = COLORS.get(node.status, COLORS["info"])
        box = FancyBboxPatch(
            (x, y),
            0.84,
            0.62,
            boxstyle="round,pad=0.025,rounding_size=0.035",
            linewidth=1.8,
            edgecolor=edge,
            facecolor=fill,
        )
        ax.add_patch(box)
        text = (
            f"{idx + 1}. {node.title}\n"
            f"{node.value}\n"
            f"{node.threshold}\n"
            f"{_wrap(node.explanation, 30)}"
        ).strip()
        ax.text(x + 0.04, y + 0.56, text, fontsize=8.8, va="top", color="#17202a")
        if node.formula_ref:
            ax.text(x + 0.04, y + 0.04, node.formula_ref, fontsize=7.4, va="bottom", color=edge)

    for idx in range(len(boxes) - 1):
        left = boxes[idx]
        right = boxes[idx + 1]
        if left["row"] == right["row"] and left["col"] < right["col"]:
            start = (left["x"] + left["w"], left["y"] + left["h"] * 0.50)
            end = (right["x"], right["y"] + right["h"] * 0.50)
            connection = "arc3,rad=0.0"
        elif left["row"] == right["row"] and left["col"] > right["col"]:
            start = (left["x"], left["y"] + left["h"] * 0.50)
            end = (right["x"] + right["w"], right["y"] + right["h"] * 0.50)
            connection = "arc3,rad=0.0"
        else:
            start = (left["x"] + left["w"] * 0.50, left["y"])
            end = (right["x"] + right["w"] * 0.50, right["y"] + right["h"])
            connection = "arc3,rad=0.10"
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color="#506070",
            shrinkA=8,
            shrinkB=8,
            connectionstyle=connection,
        )
        ax.add_patch(arrow)

    ax.text(
        0,
        0.04,
        "Дашборд построен из FuzzyXAIProofPackage: значения операторов не набраны вручную.",
        fontsize=10,
        color="#506070",
        va="bottom",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
