from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fuzzyxai.core.scenario_engine import DEFAULT_HYBRID_PLAN, compute_hybrid_xiris
from fuzzyxai.studio.operator_scenarios import load_scenarios


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_hybrid_xiris_tables(out_dir: Path) -> list[Path]:
    scenario = next(s for s in load_scenarios() if s["scenario_id"] == "hybrid_xiris")
    result = compute_hybrid_xiris()
    paths = [
        out_dir / "table_5_2_explainplan.csv",
        out_dir / "table_5_3_membership.csv",
        out_dir / "table_5_4_dE.csv",
        out_dir / "table_5_5_run_summary.csv",
    ]
    _write_csv(
        paths[0],
        [
            {"parameter": "gamma_max", "value": DEFAULT_HYBRID_PLAN.gamma_max},
            {"parameter": "delta_max", "value": DEFAULT_HYBRID_PLAN.delta_max},
            {"parameter": "theta_4", "value": DEFAULT_HYBRID_PLAN.thresholds["theta_4"]},
            {"parameter": "risk_action", "value": result.action},
        ],
    )
    _write_csv(
        paths[1],
        [
            {"field": "model_match_signal", "value": 0.88},
            {"field": "alpha_accept", "value": 0.82},
            {"field": "alpha_block", "value": 0.91},
            {"field": "u_k", "value": 0.36},
        ],
    )
    _write_csv(
        paths[2],
        [
            {"component": key, "value": value, "weight": DEFAULT_HYBRID_PLAN.gamma_weights[key]}
            for key, value in DEFAULT_HYBRID_PLAN.gamma_components.items()
        ],
    )
    summary = scenario["summary"]
    _write_csv(
        paths[3],
        [
            {"metric": "objects_total", "value": summary["objects_total"]},
            {"metric": "accept", "value": summary["accept"]},
            {"metric": "lower_confidence", "value": summary["lower_confidence"]},
            {"metric": "block", "value": summary["block"]},
            {"metric": "baseline_critical_misses", "value": summary["baseline_critical_misses"]},
            {"metric": "fuzzyxai_critical_misses", "value": summary["fuzzyxai_critical_misses"]},
            {"metric": "gamma", "value": result.gamma},
            {"metric": "delta", "value": result.delta},
            {"metric": "rho", "value": result.rho},
        ],
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["hybrid_xiris"], default="hybrid_xiris")
    parser.add_argument("--out-dir", default="reports/chapter5/studio_tables")
    args = parser.parse_args()
    paths = export_hybrid_xiris_tables(Path(args.out_dir))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
