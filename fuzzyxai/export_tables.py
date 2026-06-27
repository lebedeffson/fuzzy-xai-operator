from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fuzzyxai.core.proof_package import compute_explain_plan_hash
from fuzzyxai.core.scenario_engine import DEFAULT_HYBRID_PLAN, HybridXirisInput, compute_hybrid_xiris
from fuzzyxai.studio.operator_scenarios import load_scenarios


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
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
        out_dir / "table_5_6_risk_decomposition.csv",
    ]
    plan_hash = compute_explain_plan_hash(DEFAULT_HYBRID_PLAN.__dict__)
    _write_csv(
        paths[0],
        [{"section": "meta", "parameter": "explain_plan_version", "value": "EP-2026-01", "role": "версия ExplainPlan"}, {"section": "meta", "parameter": "explain_plan_hash", "value": plan_hash, "role": "контрольная сумма ExplainPlan"}]
        + [{"section": "alignment", "parameter": "gamma_max", "value": DEFAULT_HYBRID_PLAN.gamma_max, "role": "порог согласования"}]
        + [{"section": "alignment", "parameter": f"w_{key}", "value": value, "role": f"вес компоненты {key}"} for key, value in DEFAULT_HYBRID_PLAN.gamma_weights.items()]
        + [{"section": "reduction", "parameter": "delta_max", "value": DEFAULT_HYBRID_PLAN.delta_max, "role": "порог редукции"}]
        + [{"section": "risk", "parameter": f"w_{key}", "value": value, "role": f"вес компоненты риска {key}"} for key, value in DEFAULT_HYBRID_PLAN.risk_weights.items()]
        + [{"section": "action", "parameter": key, "value": value, "role": "порог действия"} for key, value in DEFAULT_HYBRID_PLAN.thresholds.items()]
        + [{"section": "action", "parameter": "risk_action", "value": result.action, "role": "итоговое действие"}],
    )
    inputs = HybridXirisInput()
    _write_csv(
        paths[1],
        [
            {"variable": "p_match", "term": "high", "input_value": inputs.model_match_signal, "membership_function": "mu_high(p)=p", "computed_membership": inputs.model_match_signal},
            {"variable": "Q_seg", "term": "low", "input_value": inputs.segmentation_quality, "membership_function": "mu_low(Q)=1-Q/3 rounded by ExplainPlan gate", "computed_membership": inputs.alpha_block},
            {"variable": "quality_ok", "term": "sufficient", "input_value": inputs.segmentation_quality, "membership_function": "mu_accept from adapter rule activation", "computed_membership": inputs.alpha_accept},
            {"variable": "block_condition", "term": "active", "input_value": inputs.alpha_block, "membership_function": "mu_block=alpha_block", "computed_membership": inputs.alpha_block},
            {"variable": "u_k", "term": "aggregated uncertainty", "input_value": 0.36, "membership_function": "u_k from model/quality disagreement", "computed_membership": 0.36},
        ],
    )
    definitions = {
        "d_mu": "рассогласование функций принадлежности между модельным сигналом и качеством сегментации",
        "d_R": "рассогласование состава правил принятия и блокировки",
        "d_u": "рассогласование агрегированной неопределённости",
        "d_tau": "рассогласование трассируемого следа",
    }
    _write_csv(
        paths[2],
        [
            {
                "component": key,
                "value": value,
                "weight": DEFAULT_HYBRID_PLAN.gamma_weights[key],
                "contribution": round(value * DEFAULT_HYBRID_PLAN.gamma_weights[key], 6),
                "definition": definitions.get(key, ""),
            }
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
    risk_components = {
        "model_signal": inputs.model_match_signal,
        "block_rule": inputs.alpha_block,
        "source_conflict": 1.0,
        "reduction_component": 0.3225,
    }
    risk_rows = [
        {
            "component": key,
            "value": value,
            "weight": DEFAULT_HYBRID_PLAN.risk_weights[key],
            "contribution": round(value * DEFAULT_HYBRID_PLAN.risk_weights[key], 6),
        }
        for key, value in risk_components.items()
    ]
    risk_rows.append({"component": "rho", "value": "total", "weight": "", "contribution": result.rho})
    _write_csv(paths[4], risk_rows)
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
