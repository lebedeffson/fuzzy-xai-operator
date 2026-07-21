#!/usr/bin/env python3
"""Build evidence-linked formative figures without confirmatory captions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES, load_experiment, prepare


def main() -> None:
    prepare()
    _route_figure()
    _scaling_figure()
    _grid_figure()
    print("PASS: chapter4_formative_figures figures=3")


def _route_figure() -> None:
    report = load_experiment("H5_A_route_validity")
    methods = report["methods"]
    labels = [str(row["method"]).replace("_", " ") for row in methods]
    f1 = [float(row["f1"]) for row in methods]
    false_certification = [float(row["false_certification"]) for row in methods]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(positions - 0.18, f1, width=0.36, label="Fault F1", color="#237A57")
    axis.bar(positions + 0.18, false_certification, width=0.36, label="False certification", color="#C84B31")
    axis.set_xticks(positions, labels, rotation=28, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("H5-A formative controlled route-validity comparison")
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES / "h5a_route_validity_formative.png", dpi=180)
    plt.close(figure)


def _scaling_figure() -> None:
    report = load_experiment("H9_scalability")
    rows = report["measurements"]
    sizes = [int(row["n_objects"]) for row in rows]
    wall = [float(row["wall_time_seconds"]) for row in rows]
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.loglog(sizes, wall, marker="o", color="#165D8B", linewidth=2)
    axis.set_xlabel("Objects")
    axis.set_ylabel("Operator-layer wall time, seconds")
    axis.set_title("H9 formative streaming scalability; local explainer excluded")
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / "h9_scalability_formative.png", dpi=180)
    plt.close(figure)


def _grid_figure() -> None:
    report = load_experiment("H8_grid_sensitivity")
    modalities = report["modalities"]
    configurations = [row["configuration"] for row in modalities[0]["configurations"] if row["configuration"] != "default"]
    matrix = []
    for modality in modalities:
        values = {row["configuration"]: row for row in modality["configurations"]}
        matrix.append([float(values[name]["action_agreement"]) for name in configurations])
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(matrix, vmin=0.9, vmax=1.0, cmap="YlGn")
    axis.set_xticks(range(len(configurations)), configurations)
    axis.set_yticks(range(len(modalities)), [row["modality"] for row in modalities])
    axis.set_title("H8 formative action agreement across component grids")
    figure.colorbar(image, ax=axis, label="Action agreement")
    figure.tight_layout()
    figure.savefig(FIGURES / "h8_grid_sensitivity_formative.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
