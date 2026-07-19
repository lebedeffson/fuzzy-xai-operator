from __future__ import annotations

from pathlib import Path

from fuzzyxai.evidence.contracts import TrainingObjectTrace


def render_training_trajectory(trace: TrainingObjectTrace, output_path: str | Path | None = None):
    """Show when one object was learned, destabilized, and forgotten."""

    import matplotlib.pyplot as plt

    epochs = [item.get("epoch", index) for index, item in enumerate(trace.epoch_metrics)]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(epochs, trace.confidence_by_epoch, marker="o", label="confidence", color="#176b87")
    ax.plot(epochs, trace.loss_by_epoch, marker="s", label="loss", color="#d18d24")
    for index, epoch in enumerate(trace.forgetting_events):
        ax.axvline(epoch, color="#b94a48", linestyle="--", label="forgetting event" if index == 0 else None)
    ax.set_title(f"Object {trace.object_id}: learned at epoch {trace.first_learned_epoch}, forgetting {list(trace.forgetting_events)}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Observed value")
    ax.legend()
    if output_path is None:
        return fig
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
