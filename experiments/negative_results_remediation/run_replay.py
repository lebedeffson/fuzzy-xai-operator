from __future__ import annotations

import argparse
import time

import numpy as np

from fuzzyxai.replay import CANARY_STAGES, in_canary, stream_events

from .common import ARTIFACTS, require_file, verify_protocol, write_json


def _allocate(score: np.ndarray, hard_fault: np.ndarray, budget: float) -> np.ndarray:
    # 0=accept, 1=review, 2=block. Blocks never consume review capacity.
    action = np.zeros(len(score), dtype=np.uint8)
    action[hard_fault] = 2
    available = np.flatnonzero(~hard_fault)
    capacity = min(len(available), int(budget * len(score)))
    ranked = available[np.argsort(-score[available], kind="stable")[:capacity]]
    action[ranked] = 1
    return action


def _metrics(action: np.ndarray, invalid: np.ndarray, hard_fault: np.ndarray) -> dict[str, float | int]:
    accepted = action == 0
    review = action == 1
    blocked = action == 2
    return {
        "coverage": float(np.mean(accepted)),
        "review_rate": float(np.mean(review)),
        "block_rate": float(np.mean(blocked)),
        "invalid_automatic_actions": int(np.sum(invalid & accepted)),
        "selective_operational_risk": float(np.sum(invalid & accepted) / max(1, np.sum(accepted))),
        "false_blocks": int(np.sum(blocked & ~hard_fault)),
        "false_block_rate": float(np.mean(blocked & ~hard_fault)),
        "hard_fault_recall": float(np.sum(blocked & hard_fault) / max(1, np.sum(hard_fault))),
    }


def _cluster_bootstrap(effect: np.ndarray, phase: np.ndarray, *, repetitions: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    phases = np.unique(phase)
    phase_means = np.asarray([effect[phase == item].mean() for item in phases])
    samples = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        samples[index] = rng.choice(phase_means, size=len(phases), replace=True).mean()
    estimate = float(effect.mean())
    lower, upper = np.quantile(samples, (0.025, 0.975))
    p_value = float(max(1.0 / (repetitions + 1), 2.0 * min(np.mean(samples <= 0.0), np.mean(samples >= 0.0))))
    return {"effect": estimate, "confidence_interval_95": [float(lower), float(upper)], "p_value": min(1.0, p_value), "clusters": len(phases), "repetitions": repetitions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=500_000)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    require_file(ARTIFACTS / "lock" / "negative_remediation_lock.json", "shadow replay requires frozen protocol")
    start = time.perf_counter()
    confidence = np.empty(args.events, dtype=np.float32)
    q_prediction = np.empty(args.events, dtype=np.float32)
    q_route = np.empty(args.events, dtype=np.float32)
    q_explanation = np.empty(args.events, dtype=np.float32)
    q_shift = np.empty(args.events, dtype=np.float32)
    invalid = np.empty(args.events, dtype=bool)
    hard = np.empty(args.events, dtype=bool)
    phase = np.empty(args.events, dtype=np.uint8)
    phase_names: list[str] = []
    phase_lookup: dict[str, int] = {}
    event_ids = []
    for index, event in enumerate(stream_events(args.events, seed=4201)):
        phase_id = phase_lookup.setdefault(event.phase, len(phase_lookup))
        if phase_id == len(phase_names):
            phase_names.append(event.phase)
        confidence[index] = event.confidence
        # The generator probability is used only to sample delayed labels. The
        # controller sees an observable confidence-derived estimate.
        q_prediction[index] = 1.0 - event.confidence
        q_route[index] = min(1.0, len(event.route_faults) / 2.0)
        q_explanation[index] = event.explanation_instability
        q_shift[index] = event.shift_score
        hard[index] = event.hard_fault
        phase[index] = phase_id
        invalid[index] = event.delayed_label or bool(event.route_faults) or event.explanation_instability >= 0.60 or event.shift_score >= 0.55
        if index % 1000 == 0:
            event_ids.append(event.event_id)
    accept_loss = 8.0 * q_prediction + 10.0 * q_route + 2.0 * q_explanation + 4.0 * q_shift
    review_loss = np.minimum(0.5 + 0.45 * accept_loss, 1.5 + 0.10 * accept_loss)
    review_benefit = accept_loss - review_loss
    entropy = 4.0 * confidence * (1.0 - confidence)
    simple_or = np.maximum.reduce((q_route, (q_explanation >= 0.60).astype(float), (q_shift >= 0.55).astype(float)))
    policy_scores = {
        "raw_confidence": 1.0 - confidence,
        "calibrated_confidence": q_prediction,
        "entropy_threshold": entropy,
        "conformal_selective": np.maximum(q_prediction, q_shift),
        "model_disagreement": q_shift,
        "explainer_disagreement": q_explanation,
        "simple_or": simple_or,
        "weighted_score": 0.55 * q_prediction + 0.20 * q_route + 0.15 * q_explanation + 0.10 * q_shift,
        "predictive_only": q_prediction,
        "hard_guard_only": hard.astype(float),
        "route_only": q_route,
    }
    budgets = (0.05, 0.10, 0.20, 0.30)
    budget_rows = []
    actions_by_budget = {}
    for budget in budgets:
        full = _allocate(review_benefit, hard, budget)
        policy_actions = {
            name: _allocate(score, hard if name == "hard_guard_only" else np.zeros(args.events, dtype=bool), budget)
            for name, score in policy_scores.items()
        }
        baseline = policy_actions["predictive_only"]
        full_metrics = _metrics(full, invalid, hard)
        budget_rows.append(
            {
                "review_budget": budget,
                "hierarchical_controller": full_metrics,
                "baselines": {name: _metrics(action, invalid, hard) for name, action in policy_actions.items()},
            }
        )
        actions_by_budget[budget] = (full, baseline)
    full, baseline = actions_by_budget[0.20]
    baseline_invalid_accept = invalid & (baseline == 0)
    full_invalid_accept = invalid & (full == 0)
    object_effect = baseline_invalid_accept.astype(np.int8) - full_invalid_accept.astype(np.int8)
    statistics = _cluster_bootstrap(object_effect, phase, repetitions=args.bootstrap, seed=4201)
    primary_full = _metrics(full, invalid, hard)
    primary_baseline = _metrics(baseline, invalid, hard)
    relative_reduction = (primary_baseline["invalid_automatic_actions"] - primary_full["invalid_automatic_actions"]) / max(1, primary_baseline["invalid_automatic_actions"])
    statistics["relative_reduction"] = float(relative_reduction)
    statistics["holm_adjusted_p"] = statistics["p_value"]
    criterion = bool(relative_reduction >= 0.15 and statistics["confidence_interval_95"][0] > 0.0 and statistics["holm_adjusted_p"] < 0.05 and primary_full["false_block_rate"] <= 0.01)
    phase_rows = []
    for phase_id, name in enumerate(phase_names):
        mask = phase == phase_id
        phase_rows.append({"phase": name, "events": int(mask.sum()), "hierarchical": _metrics(full[mask], invalid[mask], hard[mask]), "baseline": _metrics(baseline[mask], invalid[mask], hard[mask])})
    canary = []
    sampled_ids = [f"replay-{index:09d}" for index in range(args.events)] if args.events <= 20_000 else event_ids
    for stage in CANARY_STAGES:
        canary.append({"fraction": stage, "sampled_identity_count": len(sampled_ids), "assigned_fraction": float(np.mean([in_canary(item, stage) for item in sampled_ids]))})
    output = ARTIFACTS / "replay"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "raw_results.npz", invalid=invalid, hard_fault=hard, phase=phase, confidence=confidence, predictive_risk=q_prediction, route_risk=q_route, explanation_risk=q_explanation, shift_risk=q_shift, hierarchical_action=full, baseline_action=baseline)
    summary = {
        "phase": "preregistered_controlled_temporal_replay",
        "protocol_sha256": verify_protocol(),
        "events": args.events,
        "phase_names": phase_names,
        "primary_review_budget": 0.20,
        "budget_results": budget_rows,
        "phase_results": phase_rows,
        "statistics": statistics,
        "canary": canary,
        "elapsed_seconds": time.perf_counter() - start,
        "H3-R4": "supported_controlled_replay_only" if criterion else "not_supported",
        "production_claim_allowed": False,
        "independent_real_stream_claim_allowed": False,
        "limitation": "Incident phases and delayed labels are controlled simulation, not an observed production stream.",
    }
    write_json(output / "summary.json", summary)
    print(f"PASS remediation-shadow-replay events={args.events} H3-R4={summary['H3-R4']} production_claim=false")


if __name__ == "__main__":
    main()
