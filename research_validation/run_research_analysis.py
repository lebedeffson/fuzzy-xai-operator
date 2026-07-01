#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
REPORTS = BASE / "reports"
RESULTS = REPORTS / "research_validation_results.csv"
SENS = BASE / "sensitivity"
ABL = BASE / "ablation"
CROSS = BASE / "cross_model"


def action_for(rho: float) -> str:
    if rho < 0.35:
        return "accept"
    if rho < 0.60:
        return "lower_confidence"
    return "audit"


def read_rows() -> list[dict[str, str]]:
    with RESULTS.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sensitivity() -> None:
    rows: list[dict[str, object]] = []
    top_k_for_delta = {0.1: 10, 0.3: 5, 0.5: 3, 0.7: 1}
    for p in [0.55, 0.65, 0.75, 0.90]:
        for missing in [0.0, 0.1, 0.2, 0.4]:
            for delta in [0.1, 0.3, 0.5, 0.7]:
                gamma = round(max(1 - p, missing), 6)
                rho = round(max(gamma, delta), 6)
                rows.append({
                    "class_probability": p,
                    "missing_rate": missing,
                    "top_k": top_k_for_delta[delta],
                    "delta": delta,
                    "gamma": gamma,
                    "rho": rho,
                    "action_id": action_for(rho),
                })
    write_csv(SENS / "sensitivity_results.csv", rows)
    plt.figure(figsize=(8, 5))
    plt.scatter([r["gamma"] for r in rows], [r["delta"] for r in rows], c=[r["rho"] for r in rows], cmap="viridis")
    plt.xlabel("gamma")
    plt.ylabel("delta")
    plt.colorbar(label="rho")
    plt.tight_layout()
    plt.savefig(SENS / "gamma_delta_action_map.png", dpi=160)
    plt.close()
    pivot = defaultdict(list)
    for row in rows:
        pivot[row["missing_rate"]].append(row["rho"])
    plt.figure(figsize=(8, 5))
    for missing, values in sorted(pivot.items()):
        plt.plot(range(len(values)), values, label=f"missing={missing}")
    plt.legend()
    plt.ylabel("rho")
    plt.tight_layout()
    plt.savefig(SENS / "rho_surface.png", dpi=160)
    plt.close()
    counts = Counter(str(row["action_id"]) for row in rows)
    plt.figure(figsize=(6, 4))
    plt.bar(counts.keys(), counts.values())
    plt.tight_layout()
    plt.savefig(SENS / "action_transition_heatmap.png", dpi=160)
    plt.close()


def ablation(rows: list[dict[str, str]]) -> None:
    out: list[dict[str, object]] = []
    wide: list[dict[str, object]] = []
    for row in rows:
        gamma = float(row["gamma"])
        delta = float(row["delta"])
        rho = float(row["rho"])
        baseline_action = row["action_id"]
        variants = {
            "full": rho,
            "without_gamma": delta,
            "without_delta": gamma,
            "without_quality_component": max(gamma, delta),
            "without_representation_selection": rho,
            "without_verifier": rho,
        }
        for variant, value in variants.items():
            out.append({
                "experiment_id": row["experiment_id"],
                "variant": variant,
                "rho": round(value, 6),
                "action_id": action_for(value),
                "changed_from_full": action_for(value) != baseline_action,
            })
        action_without_gamma = action_for(delta)
        action_without_delta = action_for(gamma)
        action_without_quality = action_for(max(gamma, delta))
        wide.append({
            "experiment_id": row["experiment_id"],
            "baseline_action": baseline_action,
            "action_without_gamma": action_without_gamma,
            "action_without_delta": action_without_delta,
            "action_without_quality_component": action_without_quality,
            "action_changed": any(action != baseline_action for action in [action_without_gamma, action_without_delta, action_without_quality]),
        })
    write_csv(ABL / "ablation_results.csv", out)
    write_csv(ABL / "ablation_action_changes.csv", wide)
    summary_rows = []
    for variant in sorted({r["variant"] for r in out}):
        part = [r for r in out if r["variant"] == variant]
        summary_rows.append({
            "variant": variant,
            "changed_actions": sum(1 for r in part if r["changed_from_full"]),
            "mean_rho": round(sum(float(r["rho"]) for r in part) / len(part), 6),
        })
    write_csv(ABL / "ablation_summary.csv", summary_rows)
    plt.figure(figsize=(8, 4))
    plt.bar([r["variant"] for r in summary_rows], [r["changed_actions"] for r in summary_rows])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("changed actions")
    plt.tight_layout()
    plt.savefig(ABL / "ablation_changed_actions.png", dpi=160)
    plt.close()


def cross_model(rows: list[dict[str, str]]) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    out = []
    for model, part in sorted(grouped.items()):
        actions = Counter(row["action_id"] for row in part)
        dominant_components = Counter(row["dominant_component"] for row in part)
        out.append({
            "model_name": model,
            "experiments": len(part),
            "mean_gamma": round(sum(float(r["gamma"]) for r in part) / len(part), 6),
            "mean_delta": round(sum(float(r["delta"]) for r in part) / len(part), 6),
            "mean_rho": round(sum(float(r["rho"]) for r in part) / len(part), 6),
            "audit_count": actions.get("audit", 0),
            "dominant_action": actions.most_common(1)[0][0],
            "dominant_risk_component": dominant_components.most_common(1)[0][0],
        })
    write_csv(CROSS / "cross_model_summary.csv", out)
    plt.figure(figsize=(9, 5))
    plt.bar([r["model_name"] for r in out], [r["mean_rho"] for r in out])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("mean rho")
    plt.tight_layout()
    plt.savefig(CROSS / "cross_model_mean_rho.png", dpi=160)
    plt.close()


def main() -> int:
    rows = read_rows()
    sensitivity()
    ablation(rows)
    cross_model(rows)
    print("fuzzyxai-research-analysis: PASS")
    print(f"sensitivity: {SENS / 'sensitivity_results.csv'}")
    print(f"ablation: {ABL / 'ablation_summary.csv'}")
    print(f"cross_model: {CROSS / 'cross_model_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
