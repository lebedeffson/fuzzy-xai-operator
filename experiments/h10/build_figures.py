from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ARTIFACT_ROOT


def _save(name: str, source: pd.DataFrame, draw) -> None:
    out = ARTIFACT_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    source.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    draw(axis, source)
    figure.tight_layout()
    figure.savefig(out / f"{name}.png", dpi=300)
    figure.savefig(out / f"{name}.pdf")
    plt.close(figure)


def _bars(axis, frame: pd.DataFrame, x: str, y: str) -> None:
    axis.bar(frame[x].astype(str), frame[y], color="0.35", edgecolor="black")
    axis.set_ylabel(y.replace("_", " "))
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", color="0.85", linewidth=0.7)


def build() -> None:
    dataset = pd.read_csv(ARTIFACT_ROOT / "confirmatory" / "dataset_summary.csv")
    method = pd.read_csv(ARTIFACT_ROOT / "confirmatory" / "method_summary.csv")
    raw = pd.read_csv(ARTIFACT_ROOT / "confirmatory" / "raw_results.csv")
    replay = pd.read_csv(ARTIFACT_ROOT / "replay" / "method_summary.csv")
    tests = pd.read_json(ARTIFACT_ROOT / "confirmatory" / "statistical_tests.json")

    primary = dataset[dataset["method"].isin(("full_h10", "independent_if_else"))].copy()
    primary["dataset_method"] = primary["dataset"] + ":" + primary["method"]
    _save("01_source_localization_by_dataset", primary[["dataset_method", "source_localization_f1"]], lambda a, f: _bars(a, f, "dataset_method", "source_localization_f1"))
    _save("02_repair_set_by_dataset", primary[["dataset_method", "repair_set_f1"]], lambda a, f: _bars(a, f, "dataset_method", "repair_set_f1"))

    tradeoff = method[["method", "false_certification", "abstention_rate"]]
    _save(
        "03_false_certification_vs_abstention",
        tradeoff,
        lambda a, f: (a.scatter(f["abstention_rate"], f["false_certification"], c="0.2"), a.set_xlabel("abstention rate"), a.set_ylabel("false certification")),
    )
    known_unknown = method[["method", "leaf_f1", "unknown_recall"]]
    _save(
        "04_known_f1_vs_unknown_recall",
        known_unknown,
        lambda a, f: (a.scatter(f["leaf_f1"], f["unknown_recall"], c="0.2"), a.set_xlabel("known leaf F1"), a.set_ylabel("unknown recall")),
    )
    _save("05_minimal_cut_cost_ratio", method[["method", "cut_cost_ratio"]], lambda a, f: _bars(a, f, "method", "cut_cost_ratio"))
    _save("06_diagnostic_latency", method[["method", "diagnostic_latency"]], lambda a, f: _bars(a, f, "method", "diagnostic_latency"))

    delays = replay[["method", "detection_delay_p95", "repair_delay_p95"]].melt(id_vars="method", var_name="delay", value_name="events")
    delays["method_delay"] = delays["method"] + ":" + delays["delay"]
    _save("07_replay_delays", delays[["method_delay", "events"]], lambda a, f: _bars(a, f, "method_delay", "events"))

    effect = tests[["metric", "effect", "ci_low", "ci_high"]]
    def draw_effect(axis, frame):
        y = np.arange(len(frame))
        axis.errorbar(frame["effect"], y, xerr=[frame["effect"] - frame["ci_low"], frame["ci_high"] - frame["effect"]], fmt="o", color="black")
        axis.axvline(0.0, color="0.5", linestyle="--")
        axis.set_yticks(y, frame["metric"])
        axis.set_xlabel("H10 minus baseline")
    _save("08_effect_sizes_hierarchical_ci", effect, draw_effect)

    full_invalid = raw[(raw["method"] == "full_h10") & raw["truth_parent"].notna()]
    confusion = pd.crosstab(full_invalid["truth_parent"], full_invalid["predicted_parent"], dropna=False)
    confusion_source = confusion.reset_index().melt(id_vars="truth_parent", var_name="predicted_parent", value_name="count")
    def draw_confusion(axis, frame):
        matrix = confusion.to_numpy()
        axis.imshow(matrix, cmap="Greys")
        axis.set_xticks(range(len(confusion.columns)), confusion.columns, rotation=45, ha="right")
        axis.set_yticks(range(len(confusion.index)), confusion.index)
        axis.set_xlabel("predicted parent")
        axis.set_ylabel("truth parent")
    _save("09_parent_family_confusion", confusion_source, draw_confusion)

    full = raw[raw["method"] == "full_h10"].copy()
    curve_rows = []
    for threshold in np.linspace(float(full["anomaly_score"].min()), float(full["anomaly_score"].max()), 30):
        accept = full["anomaly_score"] <= threshold
        coverage = float(accept.mean())
        risk = float(full.loc[accept, "false_certification"].mean()) if accept.any() else 0.0
        curve_rows.append({"threshold": threshold, "coverage": coverage, "risk": risk})
    curve = pd.DataFrame(curve_rows)
    _save(
        "10_open_set_coverage_risk",
        curve,
        lambda a, f: (a.plot(f["coverage"], f["risk"], color="black"), a.set_xlabel("coverage"), a.set_ylabel("false-certification risk")),
    )


if __name__ == "__main__":
    build()
