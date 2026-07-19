from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .view_model import ExplanationViewModel


def _numeric_items(values: Mapping[str, Any]) -> tuple[list[str], list[float]]:
    labels: list[str] = []
    numbers: list[float] = []
    for key, value in values.items():
        if isinstance(value, (int, float)):
            labels.append(str(key))
            numbers.append(float(value))
    return labels, numbers


def _empty(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, color="#5f6872")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d6dde1")


def render_explanation_dashboard(
    view_model: ExplanationViewModel,
    output_path: str | Path | None = None,
):
    """Render the claim-centered story, with the v1 dashboard as fallback."""

    if view_model.visual_spec:
        from .matplotlib_renderer import render_visual_spec

        return render_visual_spec(view_model.visual_spec, view="explanation_story", output_path=output_path)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib and numpy are required for dashboard rendering") from exc

    layers = dict(view_model.layers)
    fig = plt.figure(figsize=(17, 11), constrained_layout=True, facecolor="#f5f7f6")
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 0.9))
    axes = [fig.add_subplot(grid[row, column]) for row in range(3) for column in range(2)]
    for ax in axes:
        ax.set_facecolor("white")

    # 1. Data and quality
    data_items = list(layers.get("data", []))
    if data_items:
        data = data_items[0]
        scores = {key: value for key, value in data.get("outlier_scores", {}).items() if isinstance(value, (int, float))}
        labels, values = _numeric_items(scores)
        colors = ["#b94a48" if value >= 3.5 else "#2f7187" for value in values]
        axes[0].barh(labels, values, color=colors)
        axes[0].axvline(3.5, color="#b94a48", linestyle="--", linewidth=1, label="robust deviation threshold")
        axes[0].legend(loc="lower right", fontsize=8)
        axes[0].set_xlabel("Robust distance from the reference median")
        axes[0].set_title(f"1. Data: quality {float(data.get('data_quality', 0.0)):.2f} and atypical features")
    else:
        _empty(axes[0], "Data-quality evidence was not supplied")
        axes[0].set_title("1. Data and quality")

    # 2. Training trajectory
    training_items = list(layers.get("training", []))
    if training_items:
        trace = training_items[0]
        epochs = [item.get("epoch", index) for index, item in enumerate(trace.get("epoch_metrics", []))]
        confidence = list(trace.get("confidence_by_epoch", []))
        losses = list(trace.get("loss_by_epoch", []))
        axes[1].plot(epochs, confidence, color="#176b87", marker="o", label="confidence")
        axes[1].plot(epochs, losses, color="#d18d24", marker="s", label="loss")
        for epoch in trace.get("forgetting_events", []):
            axes[1].axvline(epoch, color="#b94a48", linestyle="--", alpha=0.8)
        axes[1].legend(loc="best", fontsize=8)
        axes[1].set_xlabel("Epoch")
        axes[1].set_title(f"2. Training: object {trace.get('object_id')} trajectory and forgetting")
    else:
        _empty(axes[1], "Training history is unavailable; no forgetting claim is made")
        axes[1].set_title("2. How the model learned")

    # 3. Learned knowledge
    rules = list(layers.get("rules", []))
    if rules:
        primary = [item for item in rules if item.get("is_primary")][:7] or rules[:7]
        labels = [str(item.get("rule_id")) for item in primary]
        values = [float(item.get("importance") or item.get("coverage") or 0.0) for item in primary]
        colors = ["#2f7187" if item.get("native") else "#d18d24" for item in primary]
        axes[2].barh(labels[::-1], values[::-1], color=colors[::-1])
        axes[2].set_xlabel("Evidence-backed importance (or coverage when importance is unavailable)")
        axes[2].set_title("3. Learned knowledge: primary native and surrogate rules")
    else:
        _empty(axes[2], "This adapter exposes no auditable rules or concepts")
        axes[2].set_title("3. What the model learned")

    # 4. Current decision
    contributions = dict(view_model.model.get("contributions", {}))
    labels, values = _numeric_items(contributions)
    if values:
        colors = ["#176b87" if value >= 0 else "#b94a48" for value in values]
        axes[3].barh(labels, values, color=colors)
        axes[3].axvline(0, color="#303030", linewidth=0.8)
    else:
        score = view_model.model.get("score")
        if isinstance(score, (int, float)):
            axes[3].barh(["model score"], [float(score)], color="#176b87")
            axes[3].set_xlim(0, 1)
        else:
            _empty(axes[3], "Local contribution evidence was not supplied")
    similar_count = len(layers.get("similar_cases", []))
    counterfactual_count = len(layers.get("counterfactuals", []))
    axes[3].set_title(f"4. Current decision: {similar_count} similar cases, {counterfactual_count} tested changes")

    # 5. Trust, conflict, and action
    disagreement = dict(view_model.disagreement.get("components", {}))
    risk_components = dict(view_model.risk.get("components", {}))
    combined = {**{f"d:{key}": value for key, value in disagreement.items()}, **{f"risk:{key}": value for key, value in risk_components.items()}}
    for name in ("gamma", "delta"):
        value = view_model.disagreement.get(name)
        if isinstance(value, (int, float)):
            combined[name] = value
    labels, values = _numeric_items(combined)
    if values:
        image = axes[4].imshow([values], cmap="YlOrRd", vmin=0, vmax=max(1.0, max(values)), aspect="auto")
        axes[4].set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        axes[4].set_yticks([])
        fig.colorbar(image, ax=axes[4], fraction=0.04)
    else:
        _empty(axes[4], "Operator evidence is incomplete; automatic acceptance is not justified")
    axes[4].set_title(f"5. Trust and action: {str(view_model.risk.get('action', 'review')).upper()}")

    # Human explanation
    axes[5].set_axis_off()
    user = view_model.human_explanations.get("domain_user", view_model.human_explanations.get("user", {}))
    lines = [str(user.get("summary", view_model.narrative or "No narrative available."))]
    reasons = list(user.get("main_reasons", []))[:3]
    if reasons:
        lines.extend(
            [
                "",
                "Main reasons:",
                *[f"• {item.get('explanation', '') if isinstance(item, Mapping) else item}" for item in reasons],
            ]
        )
    limitations = list(user.get("limitations", []))[:3]
    if limitations:
        lines.extend(["", "Limits:", *[f"• {item}" for item in limitations]])
    axes[5].text(0.0, 1.0, "\n".join(lines), va="top", wrap=True, fontsize=10.5, color="#1e2a32")
    axes[5].set_title("Plain-language explanation", loc="left")

    fig.suptitle("FuzzyXAI: evidence path from data to action", fontsize=19, fontweight="bold", color="#1e2a32")
    if output_path is None:
        return fig
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
