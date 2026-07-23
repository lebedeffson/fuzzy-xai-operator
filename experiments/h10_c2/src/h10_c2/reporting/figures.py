from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_metric(path: Path, labels: list[str], values: list[float], title: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, values, color="#777777")
    axis.set_title(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)

