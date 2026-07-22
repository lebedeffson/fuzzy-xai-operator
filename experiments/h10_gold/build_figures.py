from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from .common import ARTIFACT_ROOT


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _plot(table: Path, metric: str, title: str, output_name: str) -> None:
    rows = _rows(table)
    methods = [row["method"] for row in rows]
    values = [float(row[metric]) for row in rows]
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    bars = axis.bar(methods, values, color=("#555555", "#999999", "#222222"))
    axis.set_title(f"{title} (development only)")
    axis.set_ylabel(metric.replace("_", " "))
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", color="#d0d0d0", linewidth=0.7)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    figure.tight_layout()
    output = ARTIFACT_ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{output_name}.png", dpi=300)
    figure.savefig(output / f"{output_name}.pdf")
    plt.close(figure)


def build() -> None:
    composite = ARTIFACT_ROOT / "tables" / "composite_fault_results.csv"
    _plot(composite, "source_localization_f1", "Source localization on composite routes", "source_localization_composite")
    _plot(composite, "repair_set_f1", "Repair-set agreement on composite routes", "repair_set_composite")
    _plot(composite, "cut_exact", "Minimal-cut exact match on composite routes", "minimal_cut_exact_composite")
    _plot(composite, "cut_cost_regret", "Diagnostic-cut cost regret on composite routes", "cut_cost_regret_composite")


def main() -> None:
    argparse.ArgumentParser().parse_args()
    build()


if __name__ == "__main__":
    main()
