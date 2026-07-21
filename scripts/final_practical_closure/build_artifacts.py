#!/usr/bin/env python3
"""Generate evidence-linked formative tables and figures for Chapter 4."""

from __future__ import annotations

import csv
import json

from common import FORMATIVE, ROOT, load_json, sha256, write_json


OUTPUT = ROOT / "dissertation_artifacts/final_practical_closure/chapter4_formative"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    tables = [
        _h3_table(),
        _h5_table(),
        _h6_table(),
        _h7_table(),
        _h8_table(),
        _h9_table(),
    ]
    figures = [
        _h3_figure(plt),
        _h5_figure(plt),
        _h6_figure(plt),
        _h7_figure(plt),
        _h8_figure(plt),
        _h9_figure(plt),
    ]
    manifest = {
        "schema_version": "1.0",
        "phase": "formative_development",
        "confirmatory_claim_allowed": False,
        "tables": [_entry(path) for path in tables],
        "figures": [_entry(path) for path in figures],
    }
    write_json(OUTPUT / "artifacts_manifest.json", manifest)
    print(f"PASS: practical_chapter4_formative_artifacts tables={len(tables)} figures={len(figures)} confirmatory_claim=false")


def _h3_table():
    payload = load_json(FORMATIVE / "H3_practical/summary.json")
    return _csv("h3_matched_budget.csv", payload["matched_budget_rows"])


def _h5_table():
    payload = load_json(FORMATIVE / "H5_A_route_validity/summary.json")
    return _csv("h5_route_guardrails.csv", payload["methods"])


def _h6_table():
    return _csv("h6_detectability_envelope.csv", _raw("H6_A_detectability"))


def _h7_table():
    payload = load_json(FORMATIVE / "H7_canonical_projection/summary.json")
    rows = [{"top_k": key, **values} for key, values in payload["H7_B"]["projection_tradeoff"].items()]
    return _csv("h7_projection_tradeoff.csv", rows)


def _h8_table():
    return _csv("h8_grid_sensitivity.csv", _raw("H8_grid"))


def _h9_table():
    payload = load_json(FORMATIVE / "H9_scaling/summary.json")
    return _csv("h9_scaling.csv", payload["cached_operator_layer"]["measurements"])


def _h3_figure(plt):
    payload = load_json(FORMATIVE / "H3_practical/summary.json")
    names = ("calibrated_confidence_threshold", "predictive_risk_without_route", "full_fuzzyxai_practical_controller")
    rows = [row for row in payload["matched_budget_rows"] if row["policy"] in names]
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for name in names:
        values = [row for row in rows if row["policy"] == name]
        axis.plot([row["review_budget"] for row in values], [row["wrong_or_invalid_automatic_actions"] for row in values], marker="o", label=name)
    axis.set(xlabel="Review budget", ylabel="Wrong or invalid automatic actions", title="H3 formative matched-budget comparison")
    axis.legend(fontsize=7)
    return _save(fig, "h3_matched_budget.png", plt)


def _h5_figure(plt):
    rows = load_json(FORMATIVE / "H5_A_route_validity/summary.json")["methods"]
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.barh([row["method"] for row in rows], [row["f1"] for row in rows], color="#197278")
    axis.set(xlim=(0, 1), xlabel="Fault-detection F1", title="H5-A formative controlled route validity")
    return _save(fig, "h5_route_validity.png", plt)


def _h6_figure(plt):
    rows = _raw("H6_A_detectability")
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    points = axis.scatter([row["support"] for row in rows], [row["strength"] for row in rows], c=[row["detected"] for row in rows], cmap="RdYlGn", vmin=0, vmax=1)
    axis.set(xlabel="Rule support", ylabel="Rule strength", title="H6-A formative detectability envelope")
    fig.colorbar(points, ax=axis, label="Detected")
    return _save(fig, "h6_detectability.png", plt)


def _h7_figure(plt):
    rows = load_json(FORMATIVE / "H7_canonical_projection/summary.json")["H7_B"]["projection_tradeoff"]
    keys = sorted(rows, key=int)
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.plot([int(key) for key in keys], [rows[key]["mean_retained_magnitude"] for key in keys], marker="o", color="#283d3b")
    axis.set(xlabel="Displayed reasons (top-k)", ylabel="Retained attribution magnitude", ylim=(0, 1.05), title="H7-B formative presentation trade-off")
    return _save(fig, "h7_projection_tradeoff.png", plt)


def _h8_figure(plt):
    rows = _raw("H8_grid")
    modalities = sorted({row["modality"] for row in rows})
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.bar(modalities, [sum(row["action_agreement"] for row in rows if row["modality"] == modality) / sum(row["modality"] == modality for row in rows) for modality in modalities], color="#ed9b40")
    axis.set(ylabel="Action agreement", ylim=(0, 1.05), title="H8 formative component-grid sensitivity")
    return _save(fig, "h8_grid_sensitivity.png", plt)


def _h9_figure(plt):
    rows = load_json(FORMATIVE / "H9_scaling/summary.json")["cached_operator_layer"]["measurements"]
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.loglog([row["n_objects"] for row in rows], [row["wall_time_seconds"] for row in rows], marker="o", color="#c44536")
    axis.set(xlabel="Objects", ylabel="Operator wall time, s", title="H9 formative cached operator scaling")
    return _save(fig, "h9_scaling.png", plt)


def _csv(name: str, rows):
    path = TABLES / name
    rows = list(rows)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in keys})
    return path


def _save(fig, name: str, plt):
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _entry(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "size": path.stat().st_size}


def _raw(experiment: str):
    path = FORMATIVE / experiment / "raw_results.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    main()
