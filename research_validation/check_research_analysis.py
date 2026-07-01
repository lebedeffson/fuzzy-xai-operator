#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


BASE = Path(__file__).resolve().parent
REQUIRED = [
    BASE / "sensitivity" / "sensitivity_results.csv",
    BASE / "sensitivity" / "rho_surface.png",
    BASE / "sensitivity" / "action_transition_heatmap.png",
    BASE / "sensitivity" / "gamma_delta_action_map.png",
    BASE / "ablation" / "ablation_results.csv",
    BASE / "ablation" / "ablation_action_changes.csv",
    BASE / "ablation" / "ablation_summary.csv",
    BASE / "ablation" / "ablation_changed_actions.png",
    BASE / "cross_model" / "cross_model_summary.csv",
    BASE / "cross_model" / "cross_model_mean_rho.png",
]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    errors = [f"missing {path}" for path in REQUIRED if not path.exists() or path.stat().st_size == 0]
    if not errors:
        sensitivity = rows(BASE / "sensitivity" / "sensitivity_results.csv")
        ablation = rows(BASE / "ablation" / "ablation_summary.csv")
        cross = rows(BASE / "cross_model" / "cross_model_summary.csv")
        sensitivity_fields = set(sensitivity[0])
        ablation_fields = set(rows(BASE / "ablation" / "ablation_action_changes.csv")[0])
        cross_fields = set(cross[0])
        if len(sensitivity) < 32:
            errors.append("sensitivity grid is too small")
        if not {"class_probability", "missing_rate", "top_k", "gamma", "delta", "rho", "action_id"} <= sensitivity_fields:
            errors.append("sensitivity fields are incomplete")
        if len(ablation) < 5:
            errors.append("ablation variants are missing")
        if not {"baseline_action", "action_without_gamma", "action_without_delta", "action_changed"} <= ablation_fields:
            errors.append("ablation action-change fields are incomplete")
        if len(cross) < 6:
            errors.append("cross-model coverage is too small")
        if not {"model_name", "mean_gamma", "mean_delta", "mean_rho", "dominant_action", "dominant_risk_component"} <= cross_fields:
            errors.append("cross-model fields are incomplete")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("fuzzyxai-research-analysis-check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
