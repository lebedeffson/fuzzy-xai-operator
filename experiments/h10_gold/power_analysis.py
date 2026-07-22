from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

from .common import ARTIFACT_ROOT, ROOT, load_config, write_json


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def analyze(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    rows = _rows(ARTIFACT_ROOT / "exploratory" / "development_raw_results.csv")
    population = [
        row for row in rows
        if row["case_type"] == "composite" and row["unknown"].lower() == "false"
    ]
    methods = sorted({row["method"] for row in population if row["method"] != "full_h10"})
    baseline_scores = {}
    for method in methods:
        selected = [row for row in population if row["method"] == method]
        baseline_scores[method] = sum(float(row["source_f1"]) + float(row["repair_f1"]) for row in selected) / (2 * len(selected))
    best_baseline = max(methods, key=lambda method: (baseline_scores[method], method))
    by_key: dict[tuple[str, str, str], float] = {}
    for row in population:
        if row["method"] in {"full_h10", best_baseline}:
            for metric in ("source_f1", "repair_f1"):
                by_key[(row["case_id"], row["method"], metric)] = float(row[metric])
    analyses = []
    z_alpha = NormalDist().inv_cdf(0.975)
    z_power = NormalDist().inv_cdf(0.80)
    for metric, margin_name in (("source_f1", "source_localization_macro_f1"), ("repair_f1", "repair_set_macro_f1")):
        effects = [
            by_key[(case_id, "full_h10", metric)] - by_key[(case_id, best_baseline, metric)]
            for case_id in sorted({key[0] for key in by_key if key[2] == metric})
        ]
        observed = sum(effects) / len(effects)
        variance = sum((value - observed) ** 2 for value in effects) / max(1, len(effects) - 1)
        sd = math.sqrt(variance)
        margin = float(config["registered_margins"][margin_name])
        if observed <= margin:
            required = None
            status = "STOP_EXPECTED_EFFECT_BELOW_REGISTERED_MARGIN"
        elif sd == 0.0:
            required = len(effects)
            status = "PASS"
        else:
            required = math.ceil(((z_alpha + z_power) * sd / (observed - margin)) ** 2)
            status = "PASS" if len(effects) >= required else "INCREASE_SAMPLE_BEFORE_LOCK"
        analyses.append(
            {
                "metric": metric,
                "best_baseline": best_baseline,
                "development_effect": observed,
                "registered_margin": margin,
                "paired_sd": sd,
                "development_composite_known_cases": len(effects),
                "approximate_required_cases": required,
                "status": status,
            }
        )
    result = {
        "status": "PASS" if all(row["status"] == "PASS" for row in analyses) else "BLOCKED",
        "best_baseline_selected_on_development": best_baseline,
        "baseline_scores": baseline_scores,
        "analyses": analyses,
        "sealed_test_opened": False,
        "interpretation": "Increasing sample size cannot rescue a development effect at or below the registered practical margin.",
    }
    write_json(ARTIFACT_ROOT / "exploratory" / "power_analysis.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "h10_final_gold_protocol.yaml")
    args = parser.parse_args()
    result = analyze(args.config)
    print(result["status"])


if __name__ == "__main__":
    main()
