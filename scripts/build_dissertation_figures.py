#!/usr/bin/env python3
"""Build dissertation figures from empirical evidence, never hand-entered data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "release_evidence/full_empirical_validation"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def group_policy_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine policies with identical measured coordinates before plotting."""

    grouped: dict[tuple[float, float], list[str]] = {}
    for row in rows:
        key = (float(row["automatic_coverage"]), float(row["mean_cost"]))
        grouped.setdefault(key, []).append(str(row["policy_id"]))
    return [
        {"automatic_coverage": coverage, "mean_cost": cost, "label": " / ".join(sorted(labels))}
        for (coverage, cost), labels in sorted(grouped.items())
    ]


def build(evidence: Path, output: Path) -> list[Path]:
    chapter3 = output / "chapter3"
    chapter4 = output / "chapter4"
    chapter3.mkdir(parents=True, exist_ok=True)
    chapter4.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    hierarchy = read(evidence / "uncertainty_hierarchy/hierarchy_results.json")
    rows = hierarchy["rows"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.scatter([row["mean_complexity"] for row in rows], [row["mean_risk"] for row in rows], s=80, color="#1d4e89")
    for row in rows:
        ax.annotate(row["mode"], (row["mean_complexity"], row["mean_risk"]), xytext=(5, 5), textcoords="offset points")
    ax.set(xlabel="Mean representation complexity", ylabel="Measured action risk", title="Uncertainty hierarchy: risk and complexity")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = chapter3 / "fig_3_hierarchy_risk_complexity.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    generated.append(path)

    policies = read(evidence / "policies/policy_comparison.json")
    balanced = [row for row in policies["policies"] if row["cost_scenario"] == "balanced"]
    policy_points = group_policy_points(balanced)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(
        [row["automatic_coverage"] for row in policy_points],
        [row["mean_cost"] for row in policy_points],
        marker="o",
        color="#1d4e89",
    )
    for row in policy_points:
        ax.annotate(row["label"], (row["automatic_coverage"], row["mean_cost"]), xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="Automatic coverage", ylabel="Mean predeclared cost", title="Risk-coverage curve, balanced scenario")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = chapter4 / "fig_4_risk_coverage.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    generated.append(path)

    scaling = read(evidence / "critical_rupture_scalability/critical_rupture_and_scalability.json")["scalability"]
    measurements = scaling["measurements"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.loglog([row["n_objects"] for row in measurements], [row["elapsed_seconds"] for row in measurements], marker="o", color="#a23b32")
    ax.set(xlabel="Objects", ylabel="Elapsed seconds", title="Measured explanation-graph assembly scaling")
    ax.grid(which="both", alpha=0.25)
    fig.tight_layout()
    path = chapter4 / "fig_4_scalability_loglog.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    generated.append(path)

    sensitivity = read(evidence / "sensitivity/sensitivity_report.json")["parameter_points"]
    parameters = sorted({row["parameter"] for row in sensitivity})
    factors = sorted({float(row["factor"]) for row in sensitivity})
    matrix = np.asarray([[next(row["changed_action_fraction"] for row in sensitivity if row["parameter"] == parameter and float(row["factor"]) == factor) for factor in factors] for parameter in parameters])
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=max(0.01, float(matrix.max())))
    ax.set_xticks(range(len(factors)), [f"{factor:.2f}x" for factor in factors])
    ax.set_yticks(range(len(parameters)), parameters)
    ax.set_title("Fraction of actions changed under parameter sensitivity")
    fig.colorbar(image, ax=ax, label="Changed-action fraction")
    fig.tight_layout()
    path = chapter4 / "fig_4_action_sensitivity.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    generated.append(path)

    print(f"PASS: dissertation_figures count={len(generated)}")
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=ROOT / "dissertation_artifacts")
    args = parser.parse_args()
    build(args.evidence, args.output)
