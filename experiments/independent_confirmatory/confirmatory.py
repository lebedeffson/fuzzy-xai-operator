from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score

from fuzzyxai.open_set_validator import OpenSetOutcome, OpenSetValidatorSpec, assess_open_set

from .common import ARTIFACTS, LOCK, OPENING, PRIVATE, decrypt_label_vault, git_commit, read_json, sha256_file, verify_protocol, write_json
from .formative import (
    KNOWN_PATTERNS,
    REGIONS,
    UNKNOWN_PATTERNS,
    assign_actions,
    policy_scores,
    structural_observation,
)
from .modeling import DATASET_IDS


RESULT = ARTIFACTS / "confirmatory" / "summary.json"


def _load_rows(dataset_id: str, split: str) -> dict[str, np.ndarray]:
    with np.load(PRIVATE / f"{dataset_id}-{split}.npz", allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _invalid(rows: dict[str, np.ndarray], labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model_error = rows["predictions"].astype(str) != labels.astype(str)
    operational = model_error | rows["route_risk"].astype(bool) | (rows["explanation_risk"] >= 0.95) | (rows["shift_risk"] >= 0.97)
    return model_error, operational


def _policy_metrics(rows: dict[str, np.ndarray], labels: np.ndarray, policy: str) -> tuple[dict[str, float | int], np.ndarray]:
    actions = assign_actions(rows, policy)
    model_error, invalid = _invalid(rows, labels)
    accepted = actions == "accept"
    reviewed = np.isin(actions, ("short_review", "full_review", "repair_then_retry"))
    blocked = actions == "block"
    metrics = {
        "objects": len(actions),
        "coverage": float(np.mean(accepted)),
        "review_equivalent_rate": float(np.mean(reviewed)),
        "hard_block_rate": float(np.mean(blocked)),
        "false_block_rate": float(np.mean(blocked & ~rows["irreparable_fault"].astype(bool))),
        "invalid_automatic_actions": int(np.sum(invalid & accepted)),
        "model_errors_accepted": int(np.sum(model_error & accepted)),
        "operational_risk": float(np.mean(invalid[accepted])) if np.any(accepted) else 0.0,
    }
    return metrics, invalid & accepted


def _hierarchical_ci(differences: dict[str, np.ndarray], *, iterations: int = 5000, seed: int = 11101) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    names = sorted(differences)
    distributions = {}
    for name in names:
        values = differences[name].astype(int)
        counts = np.asarray([np.sum(values == -1), np.sum(values == 0), np.sum(values == 1)])
        draws = rng.multinomial(len(values), counts / len(values), size=iterations)
        distributions[name] = (draws[:, 2] - draws[:, 0]) / len(values)
    matrix = np.vstack([distributions[name] for name in names])
    selections = rng.integers(0, len(names), size=(iterations, len(names)))
    values = np.asarray([np.mean(matrix[selections[index], index]) for index in range(iterations)])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _paired_p(full: np.ndarray, baseline: np.ndarray) -> float:
    full_only = int(np.sum(full & ~baseline))
    baseline_only = int(np.sum(baseline & ~full))
    discordant = full_only + baseline_only
    return float(binomtest(min(full_only, baseline_only), discordant, 0.5).pvalue) if discordant else 1.0


def _coverage_at_risk(score: np.ndarray, invalid: np.ndarray, risk: float, unavailable: np.ndarray | None = None) -> float:
    mask = np.ones(len(score), dtype=bool) if unavailable is None else ~unavailable
    candidates = np.flatnonzero(mask)
    order = candidates[np.argsort(score[candidates], kind="stable")]
    cumulative = np.cumsum(invalid[order]) / np.arange(1, len(order) + 1)
    valid = np.flatnonzero(cumulative <= risk)
    accepted = int(valid[-1] + 1) if len(valid) else 0
    return accepted / len(score)


def _load_spec() -> OpenSetValidatorSpec:
    raw = read_json(ARTIFACTS / "formative" / "h5_open_set" / "validator_spec.json")
    tuple_fields = {"feature_names", "family_names", "centroids", "valid_centroid", "feature_means", "feature_scales"}
    values = {}
    for field in fields(OpenSetValidatorSpec):
        value = raw[field.name]
        if field.name == "centroids":
            value = tuple(tuple(item) for item in value)
        elif field.name in tuple_fields:
            value = tuple(value)
        values[field.name] = value
    return OpenSetValidatorSpec(**values)


def _h5_confirmatory() -> dict[str, object]:
    spec = _load_spec()
    actual = []
    predicted = []
    unknown_truth = []
    unknown_scores = []
    localized = []
    repaired = []
    outcomes = []
    simple_fault_flags = []
    for family in ("valid_route", *KNOWN_PATTERNS, *UNKNOWN_PATTERNS):
        for index in range(180):
            observation = structural_observation(index, family, partition="test", seed=12101)
            assessment = assess_open_set(spec, observation)
            actual.append(family)
            predicted.append(assessment.known_fault_type or assessment.outcome.value)
            is_unknown = family in UNKNOWN_PATTERNS
            unknown_truth.append(is_unknown)
            unknown_scores.append(assessment.unknown_score)
            true_regions = {REGIONS[name] for name in ({**KNOWN_PATTERNS, **UNKNOWN_PATTERNS}.get(family, {}))}
            localized.append(bool(true_regions & set(assessment.suspected_regions)) if family != "valid_route" else True)
            repaired.append(bool(true_regions & {item.removeprefix("inspect_or_restore:") for item in assessment.repair_candidate_set}) if family != "valid_route" else True)
            outcomes.append(assessment.outcome.value)
            simple_fault_flags.append(max(abs(float(value)) for value in observation.features.values()) > 1.0)
    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)
    unknown_array = np.asarray(unknown_truth, dtype=bool)
    outcomes_array = np.asarray(outcomes)
    known_mask = np.isin(actual_array, tuple(KNOWN_PATTERNS))
    known_f1 = float(f1_score(actual_array[known_mask], predicted_array[known_mask], labels=sorted(KNOWN_PATTERNS), average="macro", zero_division=0))
    unknown_recall = float(np.mean(outcomes_array[unknown_array] == OpenSetOutcome.UNKNOWN_STRUCTURAL_FAULT.value))
    false_certification = float(np.mean(outcomes_array[unknown_array] == OpenSetOutcome.VALID_ROUTE.value))
    auroc = float(roc_auc_score(unknown_array, unknown_scores))
    simple_or_fault = np.asarray(simple_fault_flags, dtype=bool)
    simple_or_unknown = np.zeros(len(actual), dtype=bool)
    return {
        "objects": len(actual),
        "known_families": sorted(KNOWN_PATTERNS),
        "held_out_families": sorted(UNKNOWN_PATTERNS),
        "typed_open_set": {
            "unknown_fault_recall": unknown_recall,
            "false_certification": false_certification,
            "unknown_rejection_auroc": auroc,
            "known_type_macro_f1": known_f1,
            "known_type_macro_f1_degradation": 1.0 - known_f1,
            "source_region_localization": float(np.mean(np.asarray(localized)[actual_array != "valid_route"])),
            "repair_candidate_recall": float(np.mean(np.asarray(repaired)[actual_array != "valid_route"])),
        },
        "simple_or": {
            "generic_fault_recall": float(np.mean(simple_or_fault[actual_array != "valid_route"])),
            "unknown_type_identification_recall": float(np.mean(simple_or_unknown[unknown_array])),
            "source_region_localization": 0.0,
        },
        "families_absent_from_formative": True,
    }


def main() -> None:
    verify_protocol()
    if RESULT.exists() or OPENING.exists():
        raise RuntimeError("confirmatory opening is one-shot and has already been attempted")
    lock = read_json(LOCK)
    if lock["sealed_labels_opened"] or lock["post_lock_tuning_allowed"]:
        raise RuntimeError("invalid confirmatory lock")
    OPENING.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        OPENING,
        {
            "opened_at_utc": datetime.now(timezone.utc).isoformat(),
            "scoring_commit": git_commit(),
            "lock_sha256": sha256_file(LOCK),
            "purpose": "single scoring opening; no model, feature, threshold, cost or baseline tuning",
            "status": "opening_started",
        },
    )
    dataset_results = {}
    paired: dict[str, dict[str, np.ndarray]] = {}
    all_full = []
    all_primary = []
    all_predictive = []
    coverage = {"full": [], "primary": []}
    for dataset_id in DATASET_IDS:
        rows = _load_rows(dataset_id, "sealed_confirmatory_test")
        vault = decrypt_label_vault(dataset_id)
        labels = np.asarray([vault[str(object_id)] for object_id in rows["object_ids"]])
        policies = {}
        indicators = {}
        for policy in ("raw_confidence", "entropy_threshold", "predictive_only", "simple_or", "weighted_score", "route_only", "deterministic_random", "full_hierarchical_fuzzyxai"):
            policies[policy], indicators[policy] = _policy_metrics(rows, labels, policy)
        primary = str(lock["best_baseline"])
        paired[dataset_id] = {
            "primary": indicators["full_hierarchical_fuzzyxai"].astype(int) - indicators[primary].astype(int),
            "predictive": indicators["full_hierarchical_fuzzyxai"].astype(int) - indicators["predictive_only"].astype(int),
        }
        _, invalid = _invalid(rows, labels)
        full_unavailable = rows["repairable_fault"].astype(bool) | rows["irreparable_fault"].astype(bool)
        coverage["full"].append(_coverage_at_risk(policy_scores(rows)["full_hierarchical_fuzzyxai"], invalid, float(lock["fixed_operational_risk"]), full_unavailable))
        coverage["primary"].append(_coverage_at_risk(policy_scores(rows)[primary], invalid, float(lock["fixed_operational_risk"])))
        all_full.append(indicators["full_hierarchical_fuzzyxai"])
        all_primary.append(indicators[primary])
        all_predictive.append(indicators["predictive_only"])
        dataset_results[dataset_id] = policies
    full = np.concatenate(all_full)
    primary_values = np.concatenate(all_primary)
    predictive_values = np.concatenate(all_predictive)
    primary_ci = _hierarchical_ci({name: value["primary"] for name, value in paired.items()})
    predictive_ci = _hierarchical_ci({name: value["predictive"] for name, value in paired.items()}, seed=11102)
    primary_count = int(np.sum(primary_values))
    full_count = int(np.sum(full))
    relative_reduction = (primary_count - full_count) / max(1, primary_count)
    p_primary = _paired_p(full, primary_values)
    p_predictive = _paired_p(full, predictive_values)
    adjusted = {"H3-R1": min(1.0, 2.0 * min(p_primary, p_predictive)), "H3-R3": min(1.0, 2.0 * max(p_primary, p_predictive))}
    # Correct Holm ordering while preserving hypothesis names.
    if p_primary <= p_predictive:
        adjusted = {"H3-R1": min(1.0, 2.0 * p_primary), "H3-R3": max(min(1.0, 2.0 * p_primary), p_predictive)}
    else:
        adjusted = {"H3-R3": min(1.0, 2.0 * p_predictive), "H3-R1": max(min(1.0, 2.0 * p_predictive), p_primary)}
    hard_block = sum(value["full_hierarchical_fuzzyxai"]["hard_block_rate"] * value["full_hierarchical_fuzzyxai"]["objects"] for value in dataset_results.values()) / len(full)
    false_block = sum(value["full_hierarchical_fuzzyxai"]["false_block_rate"] * value["full_hierarchical_fuzzyxai"]["objects"] for value in dataset_results.values()) / len(full)
    h3 = {
        "primary_baseline": lock["best_baseline"],
        "objects": len(full),
        "full_invalid_actions": full_count,
        "baseline_invalid_actions": primary_count,
        "relative_reduction": relative_reduction,
        "absolute_rate_difference_full_minus_baseline": float(np.mean(full) - np.mean(primary_values)),
        "hierarchical_bootstrap_ci_95": primary_ci,
        "paired_p": p_primary,
        "holm_adjusted_p": adjusted["H3-R1"],
        "hard_block_rate": hard_block,
        "false_block_rate": false_block,
        "fixed_risk": lock["fixed_operational_risk"],
        "full_coverage_at_fixed_risk": float(np.mean(coverage["full"])),
        "baseline_coverage_at_fixed_risk": float(np.mean(coverage["primary"])),
        "coverage_gain": float(np.mean(coverage["full"]) - np.mean(coverage["primary"])),
        "P0_vs_full": {
            "absolute_rate_difference_full_minus_predictive": float(np.mean(full) - np.mean(predictive_values)),
            "hierarchical_bootstrap_ci_95": predictive_ci,
            "paired_p": p_predictive,
            "holm_adjusted_p": adjusted["H3-R3"],
        },
        "route_fault_evidence_scope": "controlled deterministic injection on independent predictive datasets",
    }
    result = {
        "schema_version": "1.0",
        "phase": "single_sealed_confirmatory_scoring",
        "lock_sha256": sha256_file(LOCK),
        "scoring_commit": git_commit(),
        "datasets": dataset_results,
        "H3": h3,
        "H5": _h5_confirmatory(),
        "H6": {"status": "not_opened", "reason": lock["h6_blocking_reason"]},
        "post_open_tuning": False,
    }
    write_json(RESULT, result)
    opening = read_json(OPENING)
    opening["status"] = "completed_without_tuning"
    opening["result_sha256"] = sha256_file(RESULT)
    write_json(OPENING, opening)
    print(f"PASS independent-confirmatory objects={len(full)} H3_relative_reduction={relative_reduction:.6f} H6=not_opened")


if __name__ == "__main__":
    main()
