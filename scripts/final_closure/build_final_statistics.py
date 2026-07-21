#!/usr/bin/env python3
"""Calculate frozen post-confirmatory effects, intervals and Holm corrections."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import ROOT, STUDY, load, sha256, write


OUTPUT = STUDY / "confirmatory"
SEED = 7419


def main() -> None:
    completion = STUDY / "confirmatory_completion_marker.json"
    if not completion.is_file() or load(completion).get("status") not in {
        "completed_once",
        "completed_via_declared_scoring_recovery",
    }:
        raise SystemExit("BLOCKED: final statistics require the one-shot confirmatory completion marker")
    summary = load(OUTPUT / "h3_h7_summary.json")
    protocol = load(STUDY / "protocol.json")
    actions = pd.read_parquet(ROOT / summary["raw_results"]["path"])
    h3 = _h3_statistics(actions, protocol)
    h5 = load(OUTPUT / "H5_A.json")
    h6a = load(OUTPUT / "H6_A.json")
    h6b = load(OUTPUT / "H6_B.json")
    h8 = load(OUTPUT / "H8.json")
    h9 = load(OUTPUT / "H9.json")
    raw_p = [h3["P1_vs_baseline"]["p_value"], h3["P1_vs_P0"]["p_value"]]
    if h3["fixed_risk"].get("p_value") is not None:
        raw_p.append(h3["fixed_risk"]["p_value"])
    adjusted = _holm(raw_p)
    h3["P1_vs_baseline"]["holm_adjusted_p"] = adjusted[0]
    h3["P1_vs_P0"]["holm_adjusted_p"] = adjusted[1]
    if len(adjusted) > 2:
        h3["fixed_risk"]["holm_adjusted_p"] = adjusted[2]
    payload = {
        "schema_version": "1.0",
        "phase": "sealed_confirmatory_statistics",
        "protocol_lock_sha256": summary["protocol_lock_sha256"],
        "H3": h3,
        "H5-A": h5,
        "H6-A": _h6a_statistics(h6a),
        "H6-B": h6b,
        "H7-A": summary["H7_A"],
        "H8": h8,
        "H9": h9,
        "effect_size_and_ci_required": True,
        "holm_families": protocol["holm_families"],
        "post_open_tuning": False,
        "protocol_deviation": summary.get("protocol_deviation"),
    }
    write(OUTPUT / "final_statistics.json", payload)
    print("PASS: final_confirmatory_statistics effects=true ci=true holm=true")


def _h3_statistics(actions: pd.DataFrame, protocol: dict[str, object]) -> dict[str, object]:
    baseline = protocol["primary_comparator_policy"]
    selected = actions[np.isclose(actions["review_budget"], 0.20)]
    pivot = selected.pivot(index=["dataset_id", "object_id_hash"], columns="policy", values="invalid_accept").reset_index()
    comparisons = {
        "P1_vs_baseline": (baseline, "full_fuzzyxai_P1"),
        "P1_vs_P0": ("predictive_risk_P0", "full_fuzzyxai_P1"),
    }
    results = {name: _paired_effect(pivot, left, right) | {"left": left, "right": right} for name, (left, right) in comparisons.items()}
    results["fixed_risk"] = _fixed_risk(
        actions,
        baseline,
        ceiling=protocol["primary_operational_risk_ceiling"],
        budgets=protocol["fixed_risk_operating_budgets_from_development"],
    )
    results["component_ablation"] = _component_ablation(pivot)
    results["ambiguity_strata"] = _strata(actions, baseline)
    return results


def _paired_effect(frame: pd.DataFrame, left: str, right: str) -> dict[str, object]:
    left_values = frame[left].to_numpy(bool)
    right_values = frame[right].to_numpy(bool)
    difference = left_values.astype(float) - right_values.astype(float)
    absolute = float(np.mean(difference))
    relative = float((left_values.sum() - right_values.sum()) / max(1, left_values.sum()))
    rng = np.random.default_rng(SEED)
    datasets = frame["dataset_id"].unique()
    replicates = []
    for _ in range(2000):
        sampled_datasets = rng.choice(datasets, size=len(datasets), replace=True)
        values = []
        for dataset in sampled_datasets:
            available = frame.index[frame["dataset_id"] == dataset].to_numpy()
            sampled = rng.choice(available, size=len(available), replace=True)
            values.extend(difference[sampled])
        replicates.append(float(np.mean(values)))
    discordant_left = int(np.sum(left_values & ~right_values))
    discordant_right = int(np.sum(~left_values & right_values))
    p_value = float(binomtest(discordant_left, discordant_left + discordant_right, p=0.5, alternative="greater").pvalue) if discordant_left + discordant_right else 1.0
    return {
        "absolute_risk_difference": absolute,
        "relative_invalid_action_reduction": relative,
        "confidence_interval_95": [float(np.quantile(replicates, 0.025)), float(np.quantile(replicates, 0.975))],
        "p_value": p_value,
        "n": len(frame),
        "unit_of_analysis": "sealed_test_object_clustered_within_dataset",
        "dataset_count": len(datasets),
    }


def _h6a_statistics(result: dict[str, object]) -> dict[str, object]:
    eligible = [row for row in result["rows"] if row.get("eligible")]
    detected = np.asarray([row["detected"] for row in eligible], dtype=float)
    rng = np.random.default_rng(SEED + 6)
    bootstrap = [float(np.mean(rng.choice(detected, size=len(detected), replace=True))) for _ in range(2000)]
    return {
        "eligible_region": result["eligible_region"],
        "detection_rate": float(np.mean(detected)),
        "confidence_interval_95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "false_discovery_rate": result["false_discovery_rate"],
        "sign_accuracy": result["eligible_sign_accuracy"],
        "n": len(eligible),
        "raw_result_sha256": sha256(OUTPUT / "H6_A.json"),
    }


def _fixed_risk(actions: pd.DataFrame, baseline: str, *, ceiling: float, budgets: dict[str, float | None]) -> dict[str, object]:
    policies = (baseline, "predictive_risk_P0", "full_fuzzyxai_P1")
    selected = {}
    for policy in policies:
        budget = budgets.get(policy)
        if budget is None:
            selected[policy] = None
            continue
        rows = actions[(actions["policy"] == policy) & np.isclose(actions["review_budget"], budget)].copy()
        accepted = rows["action"] == "accept"
        rows["accepted"] = accepted
        selected[policy] = rows
    if selected[baseline] is None or selected["full_fuzzyxai_P1"] is None:
        return {"risk_ceiling": ceiling, "frozen_development_budgets": budgets, "status": "not_estimable_no_development_operating_point", "p_value": None}
    left = selected[baseline].set_index(["dataset_id", "object_id_hash"])
    right = selected["full_fuzzyxai_P1"].set_index(["dataset_id", "object_id_hash"])
    aligned = left[["accepted"]].join(right[["accepted"]], lsuffix="_baseline", rsuffix="_p1")
    difference = aligned["accepted_p1"].astype(float) - aligned["accepted_baseline"].astype(float)
    effect = _bootstrap_mean_difference(aligned.reset_index(), difference.to_numpy())
    gain = float(difference.mean())
    p_value = float(binomtest(int(np.sum(difference > 0)), int(np.sum(difference != 0)), p=0.5, alternative="greater").pvalue) if np.any(difference != 0) else 1.0
    risks = {
        policy: float(rows.loc[rows["accepted"], "invalid"].mean()) if rows is not None and rows["accepted"].any() else None
        for policy, rows in selected.items()
    }
    return {
        "risk_ceiling": ceiling,
        "frozen_development_budgets": budgets,
        "observed_test_risk": risks,
        "coverage_gain_vs_baseline": gain,
        "confidence_interval_95": effect,
        "p_value": p_value,
        "target_met_before_holm": bool(gain >= 0.05 and effect[0] > 0 and risks["full_fuzzyxai_P1"] is not None and risks["full_fuzzyxai_P1"] <= ceiling),
    }


def _bootstrap_mean_difference(frame: pd.DataFrame, difference: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED + 2)
    datasets = frame["dataset_id"].unique()
    values = []
    for _ in range(2000):
        sample = []
        for dataset in rng.choice(datasets, size=len(datasets), replace=True):
            available = frame.index[frame["dataset_id"] == dataset].to_numpy()
            sample.extend(difference[rng.choice(available, size=len(available), replace=True)])
        values.append(float(np.mean(sample)))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _component_ablation(pivot: pd.DataFrame) -> dict[str, object]:
    full = pivot["full_fuzzyxai_P1"].to_numpy(bool)
    output = {}
    for column in sorted(name for name in pivot.columns if str(name).startswith("P1_minus_")):
        ablated = pivot[column].to_numpy(bool)
        output[str(column).removeprefix("P1_minus_")] = {
            "invalid_accept_change": int(np.sum(ablated) - np.sum(full)),
            "invalid_accept_rate_change": float(np.mean(ablated) - np.mean(full)),
        }
    return output


def _strata(actions: pd.DataFrame, baseline: str) -> dict[str, object]:
    features = _load_test_features()
    conditions = {
        "low_confidence": features["calibrated_confidence"] < 0.70,
        "high_confidence_disagreement": (features["calibrated_confidence"] >= 0.80) & (features["model_checkpoint_disagreement"] >= 0.15),
        "unstable_explanation": features[["seed_stability", "bootstrap_stability", "perturbation_stability"]].min(axis=1) < 0.40,
        "incomplete_provenance": features["provenance_completeness"] < 1.0,
        "shifted_object": features["label_free_shift_score"] >= 0.60,
        "rare_group": features["train_derived_rare_group_indicator"] > 0,
        "boundary_object": features["prediction_margin"] <= 0.10,
        "route_fault": features["typed_route_fault"].fillna(0) > 0,
    }
    identities = dict(zip(features["object_id_hash"], np.arange(len(features)), strict=True))
    primary = actions[np.isclose(actions["review_budget"], 0.20)]
    output = {}
    for name, mask in conditions.items():
        selected_ids = set(features.loc[mask, "object_id_hash"])
        rows = primary[primary["object_id_hash"].isin(selected_ids)]
        values = {}
        for policy in (baseline, "predictive_risk_P0", "full_fuzzyxai_P1"):
            policy_rows = rows[rows["policy"] == policy]
            values[policy] = int(policy_rows["invalid_accept"].sum())
        output[name] = {"n": len(selected_ids), "invalid_accepts": values}
    assert len(identities) == len(features)
    return output


def _load_test_features() -> pd.DataFrame:
    rows = []
    for path in sorted((OUTPUT / "features").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            row = {"object_id_hash": payload["object_id_hash"]}
            row.update(payload["predictive"])
            row.update(payload["route"])
            rows.append(row)
    return pd.DataFrame(rows)


def _holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.zeros(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


if __name__ == "__main__":
    main()
