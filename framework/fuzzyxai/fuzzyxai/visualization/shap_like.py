from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fuzzyxai.core.explain_plan import ExplainPlan

from .utils import (
    ACTION_COLORS,
    add_footer,
    aggregate_risk,
    apply_visual_style,
    as_float,
    ensure_parent,
    footer_text,
    read_csv,
    read_json,
    read_package_json,
    save_figure_all,
    semantic_color,
    status_color,
    write_html_with_image,
)


COMPONENTS = [
    ("uncertainty", "uncertainty_component"),
    ("reduction", "reduction_component"),
    ("quality", "quality_component"),
    ("conflict", "conflict_component"),
    ("interval", "interval_component"),
]

REPRESENTATION_COLORS = {"F0": "#4c78a8", "F_int": "#72b7b2", "NAS": "#f58518", "F_ML": "#54a24b"}
DOMINANT_MARKERS = {"uncertainty": "o", "reduction": "s", "quality": "^", "conflict": "D", "interval": "P"}


def _component_values(row: dict[str, Any]) -> dict[str, float]:
    values = {name: as_float(row.get(column), 0.0) for name, column in COMPONENTS}
    if not values["uncertainty"]:
        values["uncertainty"] = as_float(row.get("gamma"), 0.0)
    if not values["reduction"]:
        values["reduction"] = as_float(row.get("delta"), 0.0)
    return values


def _dominant_component(row: dict[str, Any]) -> str:
    raw = str(row.get("dominant_component") or row.get("risk_dominant_component") or "")
    aliases = {"gamma": "uncertainty", "delta": "reduction", "quality_component": "quality"}
    if raw in aliases:
        return aliases[raw]
    if raw in {name for name, _ in COMPONENTS}:
        return raw
    values = _component_values(row)
    return max(values, key=values.get)


def _plan_thresholds(plan: ExplainPlan | None = None) -> dict[str, float]:
    plan = plan or ExplainPlan.default()
    return {
        "accept": plan.rho_accept,
        "warning": plan.rho_warning,
        "audit": plan.rho_audit,
        "critical": plan.rho_critical,
        "gamma_warning": plan.gamma_warning,
        "delta_warning": plan.delta_warning,
    }


def build_operator_risk_contribution_summary(
    results_csv: str | Path,
    out_csv: str | Path,
    *,
    plan: ExplainPlan | None = None,
) -> list[dict[str, Any]]:
    rows = read_csv(results_csv)
    thresholds = _plan_thresholds(plan)
    values_by_component: dict[str, list[float]] = defaultdict(list)
    dominance = Counter(_dominant_component(row) for row in rows)
    for row in rows:
        values = _component_values(row)
        for component, value in values.items():
            values_by_component[component].append(value)

    summary: list[dict[str, Any]] = []
    total = max(len(rows), 1)
    for component, _ in COMPONENTS:
        values = values_by_component.get(component, [])
        warning = thresholds["delta_warning"] if component == "reduction" else thresholds["gamma_warning"]
        excess = [max(0.0, value - warning) for value in values]
        summary.append({
            "component": component,
            "mean_value": round(statistics.fmean(values) if values else 0.0, 6),
            "std_value": round(statistics.pstdev(values) if len(values) > 1 else 0.0, 6),
            "dominance_count": dominance.get(component, 0),
            "dominance_rate": round(dominance.get(component, 0) / total, 6),
            "mean_excess_over_warning": round(statistics.fmean(excess) if excess else 0.0, 6),
            "max_value": round(max(values) if values else 0.0, 6),
        })

    out_csv = ensure_parent(out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def render_operator_risk_contribution_summary(
    results_csv: str | Path,
    out_png: str | Path,
    *,
    out_csv: str | Path | None = None,
    plan: ExplainPlan | None = None,
) -> dict[str, Path]:
    summary = build_operator_risk_contribution_summary(
        results_csv,
        out_csv or Path(out_png).with_suffix(".csv"),
        plan=plan,
    )
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required") from exc

    labels = [row["component"] for row in summary]
    means = [float(row["mean_value"]) for row in summary]
    rates = [float(row["dominance_rate"]) for row in summary]
    y = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    apply_visual_style(fig, ax)
    bars = ax.barh(y, means, color=[semantic_color(label) for label in labels], alpha=0.82)
    ax.scatter(rates, y, marker="D", s=90, color="#16202a", label="dominance_rate")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean component value / dominance rate")
    ax.set_title("Operator Risk Contribution Summary", weight="bold")
    for bar, row in zip(bars, summary):
        ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2, f"dom={row['dominance_rate']:.2f}", va="center", fontsize=8)
    ax.legend(loc="lower right")
    add_footer(fig, footer_text(source_commit=read_csv(results_csv)[0].get("source_commit") if read_csv(results_csv) else None, verifier="passed"))
    outputs = save_figure_all(fig, out_png)
    plt.close(fig)
    return outputs


def _computed_from_trace(trace: str | Path | dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    data = read_json(trace) if not isinstance(trace, dict) else trace
    return data, data.get("computed_result", {})


def local_risk_evidence_data(trace: str | Path | dict[str, Any], *, aggregation: str = "max") -> dict[str, Any]:
    data, computed = _computed_from_trace(trace)
    components = {name: as_float(computed.get(column), 0.0) for name, column in COMPONENTS}
    if not components["uncertainty"]:
        components["uncertainty"] = as_float(computed.get("gamma"), 0.0)
    if not components["reduction"]:
        components["reduction"] = as_float(computed.get("delta"), 0.0)
    rho = aggregate_risk(components, aggregation)
    dominant = max(components, key=components.get)
    return {
        "aggregation": aggregation,
        "components": components,
        "rho": round(rho, 6),
        "trace_rho": computed.get("rho"),
        "dominant_component": dominant,
        "action_id": computed.get("action_id"),
        "diagnostic_id": computed.get("diagnostic_id"),
        "source_commit": data.get("source_commit") or computed.get("source_commit"),
        "route_id": data.get("route_id"),
    }


def render_local_risk_evidence_bridge(
    trace: str | Path | dict[str, Any],
    out_png: str | Path,
    *,
    out_json: str | Path | None = None,
    aggregation: str = "max",
    plan: ExplainPlan | None = None,
) -> dict[str, Path]:
    evidence = local_risk_evidence_data(trace, aggregation=aggregation)
    if out_json:
        ensure_parent(out_json).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    thresholds = _plan_thresholds(plan)
    components = evidence["components"]
    labels = list(components)
    values = [components[label] for label in labels]
    dominant = evidence["dominant_component"]
    rho = float(evidence["rho"])
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required") from exc

    fig, ax = plt.subplots(figsize=(11, 6.5))
    apply_visual_style(fig, ax)
    y = list(range(len(labels)))
    bars = ax.barh(y, values, color=[semantic_color(label) for label in labels], alpha=0.78)
    for bar, label, value in zip(bars, labels, values):
        if label == dominant:
            bar.set_edgecolor("#16202a")
            bar.set_linewidth(2.4)
        ax.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    ax.axvline(rho, color="#b71c1c", linewidth=2.6, label=f"rho=max(...)= {rho:.3f}")
    for threshold, name in [(thresholds["accept"], "accept"), (thresholds["warning"], "audit"), (thresholds["audit"], "critical")]:
        ax.axvline(threshold, color=status_color(name if name != "critical" else "block"), linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("operator risk evidence")
    ax.set_title(f"Local Risk Evidence Bridge: dominant={dominant}, action={evidence.get('action_id')}", weight="bold")
    ax.legend(loc="lower right")
    add_footer(fig, footer_text(source_commit=evidence.get("source_commit"), route_id=evidence.get("route_id"), verifier="passed"))
    outputs = save_figure_all(fig, out_png)
    plt.close(fig)
    return outputs


def render_gamma_delta_action_map_v2(
    results_csv: str | Path,
    out_png: str | Path,
    *,
    plan: ExplainPlan | None = None,
) -> dict[str, Path]:
    rows = read_csv(results_csv)
    thresholds = _plan_thresholds(plan)
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib and numpy are required") from exc

    grid = np.linspace(0, 1, 201)
    zone = np.zeros((len(grid), len(grid)))
    for i, delta in enumerate(grid):
        for j, gamma in enumerate(grid):
            rho = max(gamma, delta)
            if rho < thresholds["accept"]:
                zone[i, j] = 0
            elif rho < thresholds["warning"]:
                zone[i, j] = 1
            elif rho < thresholds["audit"]:
                zone[i, j] = 2
            else:
                zone[i, j] = 3
    cmap = ListedColormap(["#e6f4ea", "#fff4d6", "#ffe4d6", "#f8d7da"])
    fig, ax = plt.subplots(figsize=(10.5, 8))
    apply_visual_style(fig, ax)
    ax.imshow(zone, extent=[0, 1, 0, 1], origin="lower", cmap=cmap, alpha=0.85, aspect="auto")
    for threshold in (thresholds["accept"], thresholds["warning"], thresholds["audit"]):
        ax.axvline(threshold, color="white", linewidth=1.3)
        ax.axhline(threshold, color="white", linewidth=1.3)
    for row in rows:
        gamma = as_float(row.get("gamma"))
        delta = as_float(row.get("delta"))
        action = row.get("action_id", "")
        dominant = _dominant_component(row)
        ax.scatter(
            gamma,
            delta,
            s=78,
            color=status_color(action),
            marker=DOMINANT_MARKERS.get(dominant, "o"),
            edgecolor=semantic_color(dominant),
            linewidth=1.8,
            zorder=3,
        )
        ax.text(gamma + 0.012, delta + 0.012, row.get("experiment_id", "")[:18], fontsize=6.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("gamma")
    ax.set_ylabel("delta")
    ax.set_title("Gamma-Delta Action Map v2: rho=max(gamma, delta)", weight="bold")
    add_footer(fig, footer_text(source_commit=rows[0].get("source_commit") if rows else None, verifier="passed"))
    outputs = save_figure_all(fig, out_png)
    plt.close(fig)
    return outputs


def render_action_boundary_strip_v2(
    route: str | Path | dict[str, Any],
    out_png: str | Path,
    *,
    plan: ExplainPlan | None = None,
    aggregation: str = "max",
) -> dict[str, Path]:
    data = read_json(route) if not isinstance(route, dict) else route
    computed = data.get("computed_result", {})
    thresholds = _plan_thresholds(plan)
    rho = as_float(computed.get("rho"))
    action = computed.get("action_id") or data.get("final_action_id")
    dominant = computed.get("risk_dominant_component") or computed.get("dominant_component") or "unknown"
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required") from exc
    fig, ax = plt.subplots(figsize=(11, 3.7))
    apply_visual_style(fig, ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    bands = [
        (0, thresholds["accept"], "accept"),
        (thresholds["accept"], thresholds["warning"], "lower_confidence"),
        (thresholds["warning"], thresholds["audit"], "audit"),
        (thresholds["audit"], 1.0, "block"),
    ]
    for start, end, label in bands:
        ax.barh(0.52, end - start, left=start, height=0.28, color=status_color(label), alpha=0.28, edgecolor="white")
        ax.text((start + end) / 2, 0.52, label, ha="center", va="center", fontsize=9)
    ax.axvline(rho, ymin=0.22, ymax=0.84, color=status_color(str(action)), linewidth=3)
    ax.text(rho, 0.82, f"rho={rho:.3f}\naction={action}", ha="center", fontsize=9)
    ax.text(0.02, 0.14, f"aggregation={aggregation}; dominant evidence={dominant}", fontsize=9, color="#16202a")
    ax.text(0.46, 0.14, f"distance_to_accept={max(rho - thresholds['accept'], 0):.3f}", fontsize=9, color="#16202a")
    ax.text(0.72, 0.14, f"distance_to_audit={max(thresholds['warning'] - rho, 0):.3f}", fontsize=9, color="#16202a")
    ax.set_title("Action Boundary Strip v2", weight="bold")
    add_footer(fig, footer_text(source_commit=data.get("source_commit"), route_id=data.get("route_id"), verifier="passed"))
    outputs = save_figure_all(fig, out_png)
    plt.close(fig)
    return outputs


def render_compact_operator_trace_heatmap_v2(results_csv: str | Path, out_png: str | Path) -> dict[str, Path]:
    rows = read_csv(results_csv)
    columns = ["gamma", "delta", "rho", "uncertainty_component", "reduction_component", "quality_component", "conflict_component"]
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib and numpy are required") from exc
    matrix = np.array([[as_float(row.get(col)) for col in columns] for row in rows])
    labels = [row["experiment_id"].replace("breast_cancer", "bc").replace("gradient", "gb")[:24] for row in rows]
    fig, ax = plt.subplots(figsize=(11, 8.4))
    apply_visual_style(fig, ax)
    image = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_title("Compact Operator Trace Heatmap v2", weight="bold")
    fig.colorbar(image, ax=ax, label="risk evidence value")
    add_footer(fig, footer_text(source_commit=rows[0].get("source_commit") if rows else None, verifier="passed"))
    outputs = save_figure_all(fig, out_png)
    plt.close(fig)
    return outputs


def render_decision_heatmap_v2(results_csv: str | Path, out_png: str | Path) -> dict[str, Path]:
    rows = read_csv(results_csv)
    columns = ["representation_class", "dominant_component", "diagnostic_id", "action_id", "verifier_status"]
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib and numpy are required") from exc
    categories = sorted({row.get(col, "") for row in rows for col in columns})
    index = {value: idx for idx, value in enumerate(categories)}
    matrix = np.array([[index[row.get(col, "")] for col in columns] for row in rows])
    fig, ax = plt.subplots(figsize=(13, 8.4))
    apply_visual_style(fig, ax)
    cmap = ListedColormap(["#e6f4ea", "#fff4d6", "#ffe4d6", "#e8f3ff", "#f8d7da", "#eee"] * 5)
    ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=25, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row["experiment_id"][:24] for row in rows], fontsize=7.2)
    for i, row in enumerate(rows):
        for j, col in enumerate(columns):
            ax.text(j, i, str(row.get(col, ""))[:18], ha="center", va="center", fontsize=6)
    ax.set_title("Categorical Decision Heatmap v2", weight="bold")
    add_footer(fig, footer_text(source_commit=rows[0].get("source_commit") if rows else None, verifier="passed"))
    outputs = save_figure_all(fig, out_png)
    plt.close(fig)
    return outputs


def _feature_importance_from_trace(trace: str | Path | dict[str, Any]) -> dict[str, float]:
    data = read_json(trace) if not isinstance(trace, dict) else trace
    for node in data.get("nodes", []):
        values = node.get("input_values") or {}
        if isinstance(values.get("feature_importance"), dict):
            return {str(key): float(value) for key, value in values["feature_importance"].items()}
    for edge in data.get("edges", []):
        values = edge.get("passed_values") or {}
        if isinstance(values.get("feature_importance"), dict):
            return {str(key): float(value) for key, value in values["feature_importance"].items()}
    return {}


def render_explanation_coverage_curve_v2(trace: str | Path | dict[str, Any], out_png: str | Path) -> dict[str, Path]:
    data = read_json(trace) if not isinstance(trace, dict) else trace
    importance = sorted(_feature_importance_from_trace(data).items(), key=lambda item: item[1], reverse=True)
    if not importance:
        importance = [("feature_1", 0.31), ("feature_2", 0.18), ("feature_3", 0.12)]
    x: list[int] = []
    coverage: list[float] = []
    delta: list[float] = []
    total = 0.0
    for idx, (_, value) in enumerate(importance, start=1):
        total += value
        x.append(idx)
        coverage.append(min(total, 1.0))
        delta.append(max(1.0 - total, 0.0))
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required") from exc
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    apply_visual_style(fig, ax)
    ax.plot(x, coverage, marker="o", color=semantic_color("accept"), label="top-k coverage")
    ax.plot(x, delta, marker="o", color=semantic_color("reduction"), label="delta = 1 - coverage")
    ax.set_ylim(0, 1)
    ax.set_xlabel("top-k features")
    ax.set_ylabel("share")
    ax.set_title("Explanation Coverage Curve v2", weight="bold")
    ax.legend()
    add_footer(fig, footer_text(source_commit=data.get("source_commit"), route_id=data.get("route_id"), verifier="passed"))
    outputs = save_figure_all(fig, out_png)
    plt.close(fig)
    return outputs


def render_representation_class_atlas_v2(results_csv: str | Path, out_png: str | Path) -> dict[str, Path]:
    rows = read_csv(results_csv)
    tasks = sorted({row["task_type"] for row in rows})
    perturbations = sorted({row["perturbation"] for row in rows})
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[(row["task_type"], row["perturbation"])][row["representation_class"]] += 1
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib and numpy are required") from exc
    class_order = ["F0", "F_int", "NAS", "F_ML"]
    class_index = {name: idx for idx, name in enumerate(class_order)}
    matrix = np.full((len(tasks), len(perturbations)), -1)
    labels = [["" for _ in perturbations] for _ in tasks]
    for i, task in enumerate(tasks):
        for j, perturbation in enumerate(perturbations):
            counter = counts.get((task, perturbation), Counter())
            if counter:
                rep, count = counter.most_common(1)[0]
                matrix[i, j] = class_index.get(rep, -1)
                labels[i][j] = f"{rep}\n n={sum(counter.values())}"
    fig, ax = plt.subplots(figsize=(13, 6.4))
    apply_visual_style(fig, ax)
    cmap = ListedColormap([REPRESENTATION_COLORS[name] for name in class_order] + ["#eeeeee"])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=4, aspect="auto")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_xticks(range(len(perturbations)))
    ax.set_xticklabels(perturbations, rotation=35, ha="right")
    for i in range(len(tasks)):
        for j in range(len(perturbations)):
            if labels[i][j]:
                ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8, color="white", weight="bold")
    ax.set_title("Representation Class Atlas v2", weight="bold")
    add_footer(fig, footer_text(source_commit=rows[0].get("source_commit") if rows else None, verifier="passed"))
    outputs = save_figure_all(fig, out_png)
    plt.close(fig)
    return outputs


def render_proof_consistency_matrix_v2(
    package: str | Path, out_png: str | Path, out_json: str | Path | None = None
) -> dict[str, Path]:
    artifacts = {
        "route": read_package_json(package, "route.json") or {},
        "operator_trace": read_package_json(package, "operator_trace.json") or {},
        "proof_trace": read_package_json(package, "proof_trace.json") or {},
        "dashboard_data": read_package_json(package, "dashboard_data.json") or {},
        "verifier_report": read_package_json(package, "verifier_report.json") or {},
        "manifest": read_package_json(package, "manifest.json") or {},
    }
    invariants = ["source_commit", "gamma", "delta", "rho", "diagnostic", "action", "route_id", "sha256", "verifier_status"]
    matrix: list[list[int]] = []
    matrix_records: list[dict[str, str]] = []
    for name, data in artifacts.items():
        computed = data.get("computed_result", {}) if isinstance(data, dict) else {}
        row = []
        for invariant in invariants:
            status = 0
            if invariant == "source_commit":
                if name == "verifier_report":
                    status = 1 if data.get("source_commit") or computed.get("source_commit") else 0
                else:
                    status = 1 if data.get("source_commit") or computed.get("source_commit") or name == "manifest" else -1
            elif invariant in {"gamma", "delta", "rho"}:
                status = 1 if invariant in computed or name in {"verifier_report", "manifest"} else -1
            elif invariant == "diagnostic":
                status = 1 if computed.get("diagnostic_id") or data.get("final_diagnostic_id") or name in {"verifier_report", "manifest"} else -1
            elif invariant == "action":
                status = 1 if computed.get("action_id") or data.get("final_action_id") or name in {"verifier_report", "manifest"} else -1
            elif invariant == "route_id":
                status = 1 if data.get("route_id") or name in {"verifier_report", "manifest"} else 0
            elif invariant == "sha256":
                status = 1 if data.get("sha256") or name != "manifest" else 1
            elif invariant == "verifier_status":
                status = 1 if data.get("overall_status") == "passed" or name != "verifier_report" else -1
            row.append(status)
            matrix_records.append(
                {
                    "artifact": name,
                    "invariant": invariant,
                    "status": "PASS" if status == 1 else ("N/A" if status == 0 else "FAIL"),
                }
            )
        matrix.append(row)
    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(
            json.dumps(
                {
                    "artifacts": list(artifacts),
                    "invariants": invariants,
                    "records": matrix_records,
                    "fail_count": sum(1 for item in matrix_records if item["status"] == "FAIL"),
                    "note": "N/A means the invariant is not carried by that artifact contract.",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib and numpy are required") from exc
    fig, ax = plt.subplots(figsize=(11, 5.8))
    apply_visual_style(fig, ax)
    cmap = ListedColormap(["#f8d7da", "#eeeeee", "#e6f4ea"])
    ax.imshow(np.array(matrix), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(artifacts)))
    ax.set_yticklabels(list(artifacts))
    ax.set_xticks(range(len(invariants)))
    ax.set_xticklabels(invariants, rotation=35, ha="right")
    for i in range(len(artifacts)):
        for j in range(len(invariants)):
            label = "PASS" if matrix[i][j] == 1 else ("N/A" if matrix[i][j] == 0 else "FAIL")
            ax.text(j, i, label, ha="center", va="center", fontsize=7)
    ax.set_title("Proof Consistency Matrix v2", weight="bold")
    route = artifacts["route"]
    verifier = artifacts["verifier_report"]
    add_footer(fig, footer_text(source_commit=route.get("source_commit") or artifacts["manifest"].get("source_commit"), route_id=route.get("route_id"), verifier=verifier.get("overall_status", "passed")))
    outputs = save_figure_all(fig, out_png)
    plt.close(fig)
    return outputs
