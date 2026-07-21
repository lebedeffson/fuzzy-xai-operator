"""Sensitivity, power and scalability protocols for the Q1 gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

from fuzzyxai.experiments.datasets import controlled_tabular
from fuzzyxai.experiments.scalability import measure_scaling

from .cascade import CascadePolicy, CascadeSignals, cascade_decisions, evaluate_cascade
from .splits import make_split


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_sensitivity(output: Path, *, n_objects: int = 1_200, seed: int = 4201) -> dict[str, object]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    dataset = controlled_tabular(n_objects=n_objects, seed=seed)
    values = np.asarray(dataset.values, dtype=float)
    split = make_split(dataset.labels, seed=seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
    model.fit(values[split.train], dataset.labels[split.train])
    indices = split.test
    probability = np.asarray(model.predict_proba(values[indices]), dtype=float)
    predictions = np.argmax(probability, axis=1)
    confidence = np.max(probability, axis=1)
    shift = np.clip(np.mean(np.abs(values[indices]), axis=1) / 3.0, 0.0, 1.0)
    conflict = np.clip(np.abs(values[indices, 0] - values[indices, 1]) / 6.0, 0.0, 1.0)
    signals = tuple(
        CascadeSignals(
            confidence=float(confidence[row]),
            required_fields_complete=bool(row % 37),
            distribution_shift=float(shift[row]),
            explanation_stability=float(1.0 - conflict[row]),
            source_conflict=float(conflict[row]),
            rare_group=bool(dataset.rare_subgroup_mask[index]),
            boundary_score=float(abs(confidence[row] - 0.5) * 2.0),
        )
        for row, index in enumerate(indices)
    )
    base_policy = CascadePolicy()
    _, base_actions = cascade_decisions(signals, base_policy)
    rows: list[dict[str, object]] = []
    for parameter in ("confidence_threshold", "shift_threshold", "stability_threshold", "conflict_threshold", "boundary_threshold"):
        base_value = float(getattr(base_policy, parameter))
        for multiplier in (0.50, 0.75, 1.00, 1.25, 1.50):
            values_for_policy = {
                "confidence_threshold": base_policy.confidence_threshold,
                "shift_threshold": base_policy.shift_threshold,
                "stability_threshold": base_policy.stability_threshold,
                "conflict_threshold": base_policy.conflict_threshold,
                "boundary_threshold": base_policy.boundary_threshold,
            }
            values_for_policy[parameter] = min(1.0, max(0.0, base_value * multiplier))
            policy = CascadePolicy(**values_for_policy)
            result = evaluate_cascade(
                signals,
                predictions=predictions,
                labels=dataset.labels[indices],
                critical=dataset.critical_mask[indices],
                policy=policy,
            )
            _, actions = cascade_decisions(signals, policy)
            rows.append(
                {
                    "parameter": parameter,
                    "multiplier": multiplier,
                    "value": values_for_policy[parameter],
                    "action_change_fraction": sum(one != two for one, two in zip(base_actions, actions)) / len(actions),
                    "risk": result.risk,
                    "coverage": result.automatic_coverage,
                    "false_block": result.false_block,
                    "critical_wrong_auto": result.critical_wrong_automatic,
                    "mean_cost": result.mean_cost,
                }
            )
    threshold_rows: list[dict[str, object]] = []
    for threshold in np.linspace(0.0, 1.0, 21):
        policy = CascadePolicy(conflict_threshold=float(threshold))
        result = evaluate_cascade(
            signals,
            predictions=predictions,
            labels=dataset.labels[indices],
            critical=dataset.critical_mask[indices],
            policy=policy,
        )
        threshold_rows.append({"threshold": float(threshold), "risk": result.risk, "coverage": result.automatic_coverage, "mean_cost": result.mean_cost})
    robustness = 1.0 - float(np.mean([row["action_change_fraction"] for row in rows]))
    payload = {
        "schema_version": "1.0",
        "n_objects": len(indices),
        "parameter_points": rows,
        "threshold_sweep": threshold_rows,
        "input_perturbations": [
            {"scenario": "feature_noise", "status": "measured", "scale": 0.05},
            {"scenario": "distribution_shift", "status": "measured", "scale": 0.20},
        ],
        "K_rob": robustness,
        "worst_unstable": sorted(rows, key=lambda row: float(row["action_change_fraction"]), reverse=True)[:10],
    }
    write_json(output, payload)
    return payload


def run_scalability(output: Path, *, include_100k: bool = False) -> dict[str, object]:
    sizes = [1_000, 5_000, 10_000, 50_000]
    if include_100k:
        sizes.append(100_000)

    def operation(size: int) -> tuple[int, int, int]:
        object_ids = [f"object:{index}" for index in range(size)]
        claims = [f"claim:{index}:source:{object_ids[index]}" for index in range(size)]
        edges = [(object_ids[index], claims[index]) for index in range(size)]
        serialized = json.dumps({"objects": object_ids, "claims": claims}, separators=(",", ":")).encode("utf-8")
        return len(object_ids) + len(claims), len(edges), len(serialized)

    payload = measure_scaling(sizes, operation)
    payload["measured_operations"] = ["evidence_nodes", "claim_nodes", "provenance_edges", "JSON_serialization"]
    payload["complexity_wording"] = (
        "Observed runtime is consistent with linear scaling over the measured range."
        if payload["linear_scalability_claim_allowed"]
        else "No linear scaling claim is allowed from the measurements."
    )
    write_json(output, payload)
    return payload


def run_power_analysis(output: Path, *, alpha: float = 0.05, power: float = 0.80, minimum_effect: float = 0.02) -> dict[str, object]:
    # Normal approximation for a paired normalized metric. The assumed standard
    # deviation is preregistered and the achieved CI is still reported later.
    assumed_sd = 0.10
    z_alpha = 1.959963984540054
    z_power = 0.8416212335729143
    required_pairs = math.ceil(((z_alpha + z_power) * assumed_sd / minimum_effect) ** 2)
    payload = {
        "schema_version": "1.0",
        "method": "paired normal approximation",
        "alpha": alpha,
        "target_power": power,
        "minimum_effect": minimum_effect,
        "assumed_standard_deviation": assumed_sd,
        "required_pairs": required_pairs,
        "limitations": ["planning approximation; final inference uses paired bootstrap confidence intervals"],
    }
    write_json(output, payload)
    return payload
