from __future__ import annotations

import argparse

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import ARTIFACTS, sha256_file, write_json


def _save(figure: object, name: str) -> dict[str, object]:
    path = ARTIFACTS / "figures" / name
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {"path": str(path.relative_to(ARTIFACTS)), "sha256": sha256_file(path)}


def build() -> dict[str, object]:
    plt.rcParams.update({"font.family": "DejaVu Serif", "axes.grid": True, "grid.alpha": 0.25})
    figures = {}
    policy = pd.read_csv(ARTIFACTS / "policies" / "policy_results.csv")
    policy = policy[policy["cost_profile"] == "balanced"]
    selected = ["max_confidence", "weighted_linear", "simple_or", "predictive_risk_P0", "full_fuzzyxai", "random_matched_budget"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in selected:
        rows = policy[policy["policy"] == name].sort_values("automatic_coverage")
        ax.plot(rows["automatic_coverage"], rows["selective_risk"], marker="o", label=name)
    ax.set(xlabel="Automatic coverage", ylabel="Selective risk", title="Matched-coverage policy comparison")
    ax.legend(fontsize=7)
    figures["policy_risk_coverage"] = _save(fig, "policy_risk_coverage.png")

    route = pd.read_csv(ARTIFACTS / "route_faults" / "summary.csv")
    pivot = route.pivot(index="group", columns="method", values="f1")
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, color=["0.15", "0.35", "0.55", "0.75"])
    ax.set(xlabel="Fault group", ylabel="F1", ylim=(0, 1.05), title="Route-fault detection by method")
    ax.legend(fontsize=7)
    figures["route_faults"] = _save(fig, "route_faults.png")

    runtime = pd.read_csv(ARTIFACTS / "runtime" / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for axis, method in zip(axes, sorted(runtime["explainer"].unique()), strict=True):
        rows = runtime[runtime["explainer"] == method].sort_values("n")
        axis.stackplot(
            rows["n"],
            rows["model_seconds_median"],
            rows["explainer_seconds_median"],
            rows["fuzzyxai_seconds_median"],
            rows["serialization_seconds_median"],
            labels=["model", "explainer", "FuzzyXAI", "serialization"],
            colors=["0.85", "0.55", "0.25", "0.05"],
        )
        axis.set_xscale("log")
        axis.set_title(method)
        axis.set_xlabel("N")
    axes[0].set_ylabel("Median seconds")
    axes[1].legend(fontsize=7)
    figures["runtime_decomposition"] = _save(fig, "runtime_decomposition.png")

    explanations = pd.read_json(ARTIFACTS / "explanations" / "sealed_test.jsonl", lines=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(
        [explanations["ig_deletion_fidelity"], explanations["ig_perturbation_stability"], explanations["explainer_top_k_agreement"]],
        tick_labels=["Deletion fidelity", "Perturbation stability", "IG/masking agreement"],
        patch_artist=True,
        boxprops={"facecolor": "0.8"},
    )
    ax.set(ylim=(0, 1), title="Local explanation measurements on AG News")
    figures["explanation_quality"] = _save(fig, "explanation_quality.png")
    write_json(ARTIFACTS / "manifests" / "figures_manifest.json", figures)
    return {"figures": len(figures)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = build()
    print(f"PASS: figures={result['figures']}")


if __name__ == "__main__":
    main()
