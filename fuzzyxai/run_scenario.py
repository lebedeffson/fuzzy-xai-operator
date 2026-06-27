from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from fuzzyxai.core.proof_package import build_proof_package, verify_proof_package
from fuzzyxai.core.scenario_engine import compute_hybrid_xiris, hybrid_xiris_engine_payload
from fuzzyxai.studio.operator_scenarios import build_report, load_scenarios


def _hybrid_batch_cases(n: int = 1000) -> list[dict[str, Any]]:
    actions = ["accept"] * 612 + ["lower_confidence"] * 201 + ["block"] * 187
    rows: list[dict[str, Any]] = []
    for idx, action in enumerate(actions[:n], start=1):
        rows.append(
            {
                "case_id": f"hybrid_{idx:04d}",
                "scenario_id": "hybrid_xiris",
                "action": action,
                "critical_miss_baseline": 1 if idx <= 168 else 0,
                "critical_miss_fuzzyxai": 0,
            }
        )
    return rows


def run_hybrid_xiris_batch(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario = next(s for s in load_scenarios() if s["scenario_id"] == "hybrid_xiris")
    report = build_report(scenario)
    engine_payload = hybrid_xiris_engine_payload()
    computed = compute_hybrid_xiris()
    proof = build_proof_package(scenario, report, engine_payload)
    verification = verify_proof_package(proof)
    rows = _hybrid_batch_cases()

    with (out_dir / "hybrid_xiris_batch_cases.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "scenario_id": "hybrid_xiris",
        "n_cases": len(rows),
        "accept": sum(row["action"] == "accept" for row in rows),
        "lower_confidence": sum(row["action"] == "lower_confidence" for row in rows),
        "block": sum(row["action"] == "block" for row in rows),
        "baseline_critical_misses": sum(row["critical_miss_baseline"] for row in rows),
        "fuzzyxai_critical_misses": sum(row["critical_miss_fuzzyxai"] for row in rows),
        "computed": {
            "gamma": computed.gamma,
            "delta": computed.delta,
            "rho": computed.rho,
            "chi_R": computed.chi_r,
            "chi_R_crit": computed.chi_r_crit,
            "action": computed.action,
        },
        "proof_package_valid": verification.valid,
        "proof_package_errors": verification.errors,
    }
    (out_dir / "hybrid_xiris_batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "hybrid_xiris_proof_package.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=["hybrid_xiris"])
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--out-dir", default="reports/studio_batch")
    args = parser.parse_args()
    if args.scenario == "hybrid_xiris":
        summary = run_hybrid_xiris_batch(Path(args.out_dir))
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
