#!/usr/bin/env python3
"""Build claim-safe Q1 figures from measured JSON evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "release_evidence/q1_remediation"
CH3 = ROOT / "dissertation_artifacts/q1/chapter3"
CH4 = ROOT / "dissertation_artifacts/q1/chapter4"


def load(path: str) -> dict[str, object]:
    return json.loads((EVIDENCE / path).read_text(encoding="utf-8"))


def save(fig: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    getattr(fig, "tight_layout")()
    getattr(fig, "savefig")(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    h3 = load("cascade/h3_adaptive_cascade.json")
    rows = list(h3["policies"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        ax.scatter(row["automatic_coverage"], row["risk"], s=65)
        ax.annotate(row["policy_id"], (row["automatic_coverage"], row["risk"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Automatic coverage", ylabel="Predeclared decision risk", title="Q1 adaptive cascade: risk and coverage")
    ax.grid(alpha=0.25)
    save(fig, CH4 / "fig_q1_cascade_risk_coverage.png")

    h4 = load("uncertainty/h4_uncertainty_hierarchy.json")
    rows = list(h4["rows"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([row["mean_complexity"] for row in rows], [row["mean_risk"] for row in rows], s=70)
    for row in rows:
        ax.annotate(row["mode"], (row["mean_complexity"], row["mean_risk"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Representation complexity", ylabel="Controlled action risk", title="Q1 uncertainty hierarchy")
    ax.grid(alpha=0.25)
    save(fig, CH3 / "fig_q1_uncertainty_risk_complexity.png")

    h6 = load("rule_ablation/h6_rule_ablation.json")
    effects = np.asarray([row["specific_effect"] for row in h6["pairs"]], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(effects, bins=15, color="#356b8c", edgecolor="white")
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set(xlabel="Selected minus matched subgroup-recall effect", ylabel="Paired runs", title="Q1 controlled matched rule ablation")
    ax.text(0.02, 0.95, "Controlled candidate; independent confirmation pending", transform=ax.transAxes, va="top", fontsize=9)
    save(fig, CH4 / "fig_q1_rule_ablation_effect.png")

    h5 = load("critical_rupture/h5_critical_rupture.json")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(["precision", "recall", "F1"], [h5["structural"]["precision"], h5["structural"]["recall"], h5["structural"]["f1"]], color="#356b8c")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Structural diagnosis")
    axes[1].bar(["M0", "M1"], [h5["predictive"]["m0_auprc"], h5["predictive"]["m1_auprc"]], color=["#7b8b8e", "#b75d4a"])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Error prediction (AUPRC)")
    fig.suptitle("Critical rupture: structural and predictive roles are separate")
    save(fig, CH4 / "fig_q1_critical_rupture_roles.png")

    sensitivity = load("sensitivity/sensitivity.json")
    points = list(sensitivity["parameter_points"])
    parameters = sorted({row["parameter"] for row in points})
    multipliers = sorted({float(row["multiplier"]) for row in points})
    matrix = np.asarray(
        [[next(float(row["action_change_fraction"]) for row in points if row["parameter"] == parameter and float(row["multiplier"]) == multiplier) for multiplier in multipliers] for parameter in parameters]
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=max(0.01, float(matrix.max())))
    ax.set_xticks(range(len(multipliers)), [f"{item:.2f}x" for item in multipliers])
    ax.set_yticks(range(len(parameters)), parameters)
    ax.set_title("Action changes under parameter sensitivity")
    fig.colorbar(image, ax=ax, label="Changed-action fraction")
    save(fig, CH4 / "fig_q1_sensitivity_heatmap.png")

    sweep = list(sensitivity["threshold_sweep"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([row["mean_cost"] for row in sweep], [row["risk"] for row in sweep], marker="o", markersize=3)
    ax.set(xlabel="Mean cascade cost", ylabel="Predeclared decision risk", title="Q1 risk-cost frontier")
    ax.grid(alpha=0.25)
    save(fig, CH4 / "fig_q1_pareto_frontier.png")

    scaling = load("scalability/scalability.json")
    measurements = list(scaling["measurements"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog([row["n_objects"] for row in measurements], [row["elapsed_seconds"] for row in measurements], marker="o")
    ax.set(xlabel="Objects", ylabel="Elapsed seconds", title="Measured evidence-graph scaling")
    ax.grid(alpha=0.25, which="both")
    save(fig, CH4 / "fig_q1_scalability.png")
    print("PASS: q1_figures")


if __name__ == "__main__":
    main()
