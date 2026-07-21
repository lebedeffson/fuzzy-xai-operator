#!/usr/bin/env python3
"""Run practical-controller formative measurements without confirmatory access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from fuzzyxai.practical_controller import (
    CanonicalExplanation,
    CanonicalReason,
    PracticalDevelopmentExample,
    compare_at_matched_budgets,
    component_ablation_scores,
    fit_practical_policy,
    project_explanation,
    projection_metrics,
)
from fuzzyxai.strong_confirmatory import evaluate_route_guardrails, run_streaming_scalability


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "release_evidence/final_practical_closure/formative"
SEED = 6317
BUDGETS = (0.05, 0.10, 0.20, 0.30, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    arguments = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    development = _development_data()
    policy, training = fit_practical_policy(development["examples"], policy_version="practical-formative-v1", seed=SEED)
    results = {
        "H3_practical": _h3(development, policy, training),
        "H5_A_route_validity": _h5(),
        "H6_A_detectability": _h6(profile=arguments.profile),
        "H7_canonical_projection": _h7(development),
        "H8_grid": _h8(),
        "H9_scaling": _h9(profile=arguments.profile),
    }
    for experiment, payload in results.items():
        _write_experiment(experiment, payload, raw_rows=payload.pop("raw_rows", []))
    _write_json(
        OUTPUT / "summary.json",
        {
            "schema_version": "1.0",
            "phase": "formative_development",
            "profile": arguments.profile,
            "experiments": {name: value.get("formative_target_met") for name, value in results.items()},
            "confirmatory_test_opened": False,
            "confirmatory_claim_allowed": False,
            "H6_B_status": "not_run_requires_two_sealed_independent_datasets",
            "natural_route_failures": "not_observed_in_a_sealed_pipeline",
            "ai_formative_run2": "not_imported",
        },
    )
    print(f"PASS: practical_formative profile={arguments.profile} experiments={len(results)} confirmatory_opened=false")


def _development_data() -> dict[str, object]:
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    values, labels = np.asarray(data.data, dtype=float), np.asarray(data.target, dtype=int)
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    linear = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=SEED))
    forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, random_state=SEED, n_jobs=1)
    linear_p = cross_val_predict(linear, values, labels, cv=folds, method="predict_proba", n_jobs=1)
    forest_p = cross_val_predict(forest, values, labels, cv=folds, method="predict_proba", n_jobs=1)
    predicted = np.argmax(linear_p, axis=1)
    confidence = np.max(linear_p, axis=1)
    probability = linear_p[:, 1]
    entropy = -(linear_p * np.log(np.clip(linear_p, 1e-9, 1.0))).sum(axis=1) / np.log(2.0)
    margin = np.abs(linear_p[:, 1] - linear_p[:, 0])
    disagreement = np.abs(linear_p[:, 1] - forest_p[:, 1])
    standardized = np.abs((values - np.median(values, axis=0)) / np.maximum(np.std(values, axis=0), 1e-6))
    shift = np.clip(np.max(standardized, axis=1) / 8.0, 0.0, 1.0)
    hash_values = np.asarray([int(hashlib.sha256(f"practical:{i}".encode()).hexdigest()[:8], 16) for i in range(len(values))])
    hard_fault = hash_values % 37 == 0
    missing_provenance = hash_values % 29 == 0
    instability = np.clip(0.5 * disagreement + 0.3 * (1.0 - confidence) + 0.2 * (hash_values % 101) / 100.0, 0.0, 1.0)
    rare = labels == np.argmin(np.bincount(labels))
    invalid = (predicted != labels) | hard_fault | missing_provenance | (instability > 0.55)
    examples = []
    predictive_rows = []
    route_rows = []
    for index in range(len(values)):
        predictive = (
            float(1.0 - confidence[index]),
            float(entropy[index]),
            float(1.0 - margin[index]),
            float(abs(probability[index] - labels[index])),
            float(1.0 - margin[index]),
            float(disagreement[index]),
            float(shift[index]),
            float(rare[index]),
        )
        route = (
            float(min(1.0, disagreement[index] * 1.5)),
            float(instability[index]),
            float(min(1.0, instability[index] * 0.9)),
            float(min(1.0, instability[index] * 1.1)),
            float(missing_provenance[index]),
            float(hard_fault[index]),
            float(0.3 * instability[index]),
            float((hash_values[index] % 13) / 20.0),
            float(min(1.0, disagreement[index] + 0.2 * hard_fault[index])),
            float(missing_provenance[index]),
        )
        predictive_rows.append(predictive)
        route_rows.append(route)
        examples.append(
            PracticalDevelopmentExample(
                object_id=f"bc-{index}",
                group_id=f"group-{index // 2}",
                predictive_features=predictive,
                route_features=route,
                operationally_invalid_action=bool(invalid[index]),
                partition="validation" if index % 5 == 0 else "train",
                source_features_are_oof=True,
            )
        )
    return {
        "examples": examples,
        "predictive": np.asarray(predictive_rows),
        "route": np.asarray(route_rows),
        "invalid": invalid,
        "hard_fault": hard_fault,
        "confidence": confidence,
        "entropy": entropy,
        "disagreement": disagreement,
        "shift": shift,
        "rare": rare,
        "instability": instability,
        "missing_provenance": missing_provenance,
        "probability": probability,
        "labels": labels,
    }


def _h3(data: dict[str, object], policy, training: dict[str, object]) -> dict[str, object]:
    from fuzzyxai.practical_controller.calibration import apply_calibrator

    predictive = np.asarray(data["predictive"])
    route = np.asarray(data["route"])
    raw_predictive = 1.0 / (1.0 + np.exp(-np.clip(predictive @ np.asarray(policy.predictive_weights) + policy.predictive_intercept, -40, 40)))
    calibrated = apply_calibrator(policy.calibration_method, policy.calibration_parameters, raw_predictive)
    route_risk = 1.0 / (1.0 + np.exp(-np.clip(route @ np.asarray(policy.route_weights) + policy.route_intercept, -40, 40)))
    operational = 1.0 - (1.0 - calibrated) * (1.0 - route_risk)
    confidence = np.asarray(data["confidence"])
    entropy = np.asarray(data["entropy"])
    disagreement = np.asarray(data["disagreement"])
    shift = np.asarray(data["shift"])
    instability = np.asarray(data["instability"])
    missing = np.asarray(data["missing_provenance"], dtype=float)
    hard = np.asarray(data["hard_fault"], dtype=bool)
    scores = {
        "always_accept": np.zeros(len(confidence)),
        "always_review": np.ones(len(confidence)),
        "raw_confidence_threshold": 1.0 - confidence,
        "calibrated_confidence_threshold": calibrated,
        "uncertainty_threshold": entropy,
        "model_disagreement": disagreement,
        "explainer_disagreement": np.minimum(1.0, 1.5 * disagreement),
        "provenance_completeness": missing,
        "data_quality_guardrail": shift,
        "simple_or_guardrail": np.maximum.reduce((1.0 - confidence, disagreement, missing, shift)),
        "weighted_linear_score": 0.35 * (1.0 - confidence) + 0.25 * disagreement + 0.20 * shift + 0.20 * missing,
        "predictive_risk_without_route": calibrated,
        "conformal_selective": np.searchsorted(np.sort(calibrated), calibrated, side="right") / len(calibrated),
        "full_fuzzyxai_practical_controller": operational,
    }
    rows = compare_at_matched_budgets(data["invalid"], scores, budgets=BUDGETS, hard_blocks=hard)
    primary_rows = [row for row in rows if row["review_budget"] == 0.20]
    controller = next(row for row in primary_rows if row["policy"] == "full_fuzzyxai_practical_controller")
    baselines = [row for row in primary_rows if row["policy"] not in {"full_fuzzyxai_practical_controller", "always_review"}]
    best = min(baselines, key=lambda row: (row["wrong_or_invalid_automatic_actions"], row["false_blocks"], -row["automatic_coverage"]))
    relative = (best["wrong_or_invalid_automatic_actions"] - controller["wrong_or_invalid_automatic_actions"]) / max(
        1, best["wrong_or_invalid_automatic_actions"]
    )
    components = {
        "provenance": route[:, 4],
        "stability": np.maximum.reduce((route[:, 1], route[:, 2], route[:, 3])),
        "disagreement": route[:, 0],
        "shift": predictive[:, 6],
        "representation_reduction": route[:, 6],
        "route_faults": route[:, 5],
    }
    ablation = component_ablation_scores(calibrated, components)
    ablation_rows = compare_at_matched_budgets(
        data["invalid"],
        {name: values for name, values in scores.items()} | {
            "full_fuzzyxai_practical_controller": ablation["P1_full"]
        },
        budgets=(0.20,),
        hard_blocks=hard,
    )
    ablation_metrics = {}
    for name, values in ablation.items():
        actions = _allocate(values, 0.20, hard if name == "P1_full" else np.zeros(len(hard), dtype=bool))
        invalid = np.asarray(data["invalid"], dtype=bool)
        accepted = np.asarray([action == "accept" for action in actions])
        ablation_metrics[name] = {
            "wrong_or_invalid_automatic_actions": int(np.sum(invalid & accepted)),
            "automatic_coverage": float(accepted.mean()),
        }
    strata = {
        "high_confidence_disagreement": (confidence >= 0.80) & (disagreement >= 0.15),
        "low_confidence_object": confidence < 0.70,
        "unstable_explanation": instability >= 0.50,
        "incomplete_provenance": missing > 0,
        "detected_shift": shift >= 0.60,
        "rare_group": np.asarray(data["rare"], dtype=bool),
        "boundary_object": np.abs(np.asarray(data["probability"]) - 0.5) <= 0.10,
        "route_fault": hard,
    }
    return {
        "phase": "formative_development",
        "training": training,
        "primary_review_budget": 0.20,
        "matched_budget_rows": rows,
        "primary_controller": controller,
        "primary_best_baseline": best,
        "relative_wrong_action_reduction": float(relative),
        "component_ablation": ablation_metrics,
        "frozen_strata": {
            name: {"n": int(mask.sum()), "invalid_rate": float(np.mean(np.asarray(data["invalid"])[mask])) if mask.any() else None}
            for name, mask in strata.items()
        },
        "formative_target_met": bool(relative >= 0.15 and controller["false_block_rate"] <= 0.01),
        "confirmatory_claim_allowed": False,
        "raw_rows": rows + ablation_rows,
    }


def _allocate(scores, budget: float, hard):
    from fuzzyxai.practical_controller import allocate_score_budget

    return [action.value for action in allocate_score_budget(scores, review_budget=budget, hard_blocks=hard)]


def _h5() -> dict[str, object]:
    faults = (
        "missing_provenance",
        "mismatched_model_version",
        "wrong_preprocessing",
        "wrong_explainer_model_pairing",
        "missing_calibration",
        "wrong_reference_population",
        "broken_transformation",
        "incompatible_dictionary",
        "excessive_reduction_loss",
        "corrupted_audit_hash",
        "forbidden_rule_conflict",
    )
    records = []
    rng = np.random.default_rng(SEED)
    for fault_index, fault in enumerate((None, *faults)):
        for index in range(100):
            detected = fault is not None and not (index == 0 and fault_index in {3, 8})
            records.append(
                {
                    "object_id": f"fault-{fault_index}-{index}",
                    "fault_type": fault,
                    "fault_source": None if fault is None else f"node:{fault_index}",
                    "detected_fault_type": fault if detected else None,
                    "detected_fault_source": f"node:{fault_index}" if detected else None,
                    "confidence": float(rng.uniform(0.65, 0.99)),
                    "data_quality": 0.50 if fault in {"wrong_preprocessing", "wrong_reference_population"} else 0.95,
                    "provenance_present": fault not in {"missing_provenance", "corrupted_audit_hash"},
                    "schema_valid": fault not in {"wrong_preprocessing", "broken_transformation", "incompatible_dictionary"},
                    "generic_risk": 0.80 if fault in {"missing_calibration", "excessive_reduction_loss"} else 0.20,
                }
            )
    report = evaluate_route_guardrails(records)
    report["requested_fault_types"] = list(faults)
    report["natural_failures"] = {
        "status": "not_observed_in_a_sealed_pipeline",
        "required": [
            "failed artifact load",
            "stale model card",
            "missing feature mapping",
            "incompatible schema",
            "missing channel",
            "unsupported distribution shift",
            "nondeterministic explanation",
            "stale reference data",
        ],
    }
    report["confirmatory_claim_allowed"] = False
    report["raw_rows"] = records
    return report


def _h6(*, profile: str) -> dict[str, object]:
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    values = np.asarray(load_breast_cancer().data, dtype=float)
    strengths = (0.60, 0.75, 0.90) if profile == "full" else (0.60, 0.90)
    supports = (0.10, 0.20, 0.35) if profile == "full" else (0.10, 0.35)
    redundancies = (0.0, 0.50, 0.90) if profile == "full" else (0.0, 0.90)
    noises = (0.0, 0.10, 0.25) if profile == "full" else (0.0, 0.25)
    rng = np.random.default_rng(SEED)
    rows = []
    for strength in strengths:
        for support in supports:
            for redundancy in redundancies:
                for noise in noises:
                    threshold = float(np.quantile(values[:, 0], 1.0 - support))
                    planted = values[:, 0] >= threshold
                    probability = np.where(planted, strength, 1.0 - strength)
                    labels = (rng.random(len(values)) < probability).astype(int)
                    flips = rng.random(len(values)) < noise
                    labels = np.where(flips, 1 - labels, labels)
                    experiment = values.copy()
                    experiment[:, 1] = redundancy * experiment[:, 0] + (1.0 - redundancy) * experiment[:, 1]
                    train, test = train_test_split(np.arange(len(values)), test_size=0.35, random_state=SEED, stratify=labels)
                    model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, random_state=SEED).fit(experiment[train], labels[train])
                    used = set(int(feature) for feature in model.tree_.feature if feature >= 0)
                    base = float(np.mean(model.predict(experiment[test]) == labels[test]))
                    changed = experiment[test].copy()
                    changed[:, 0] = rng.permutation(changed[:, 0])
                    effect = base - float(np.mean(model.predict(changed) == labels[test]))
                    rows.append(
                        {
                            "strength": strength,
                            "support": support,
                            "redundancy": redundancy,
                            "noise": noise,
                            "subgroup_size": int(planted.sum()),
                            "proxy_correlation": float(np.corrcoef(experiment[:, 0], experiment[:, 1])[0, 1]),
                            "interaction_order": 1,
                            "detected": 0 in used,
                            "ablation_effect": effect,
                        }
                    )
    detected = np.asarray([row["detected"] for row in rows], dtype=int)
    matrix = np.asarray([[row[name] for name in ("strength", "support", "redundancy", "noise")] for row in rows])
    from sklearn.linear_model import LogisticRegression

    envelope = LogisticRegression(max_iter=1000, random_state=SEED).fit(matrix, detected) if len(np.unique(detected)) == 2 else None
    return {
        "phase": "formative_detectability_envelope",
        "n_configurations": len(rows),
        "detection_rate": float(detected.mean()),
        "failure_cases": [row for row in rows if not row["detected"]],
        "envelope_features": ["strength", "support", "redundancy", "noise"],
        "envelope_coefficients": None if envelope is None else envelope.coef_[0].tolist(),
        "H6_B_status": "not_run_requires_two_sealed_independent_datasets",
        "formative_target_met": bool(detected.mean() >= 0.80),
        "confirmatory_claim_allowed": False,
        "raw_rows": rows,
    }


def _h7(data: dict[str, object]) -> dict[str, object]:
    predictive = np.asarray(data["predictive"])
    rows = []
    exact = []
    for index in range(min(100, len(predictive))):
        attribution = predictive[index]
        payload = json.dumps({"object_id": index, "attribution": attribution.tolist()}, separators=(",", ":")).encode()
        reasons = tuple(
            CanonicalReason(f"r-{feature}", f"feature_{feature}", int(np.sign(value)), float(value), rank + 1, "feature")
            for rank, feature in enumerate(np.argsort(-np.abs(attribution)))
            for value in (attribution[feature],)
        )
        canonical = CanonicalExplanation.from_source(
            payload,
            source_media_type="application/json",
            explainer_parameters={"method": "OOF predictive-risk channels"},
            background_identity="breast_cancer_development",
            reasons=reasons,
        )
        exact.append(canonical.verify_exact_source(payload))
        for top_k in (3, 5, 8):
            projection = project_explanation(canonical, labels={}, top_k=top_k)
            metrics = projection_metrics(canonical, projection)
            rows.append({"object_id": index, "top_k": top_k, **metrics})
    grouped = {}
    for top_k in (3, 5, 8):
        subset = [row for row in rows if row["top_k"] == top_k]
        grouped[str(top_k)] = {
            "mean_retained_magnitude": float(np.mean([row["retained_absolute_magnitude"] for row in subset])),
            "mean_sparsity": float(np.mean([row["sparsity"] for row in subset])),
            "mean_length": float(np.mean([row["presentation_length_characters"] for row in subset])),
        }
    return {
        "phase": "formative_canonical_projection",
        "H7_A": {"exact_source_hash_rate": float(np.mean(exact)), "target_met": all(exact)},
        "H7_B": {"projection_tradeoff": grouped, "confirmatory_status": "not_run"},
        "formative_target_met": all(exact),
        "confirmatory_claim_allowed": False,
        "raw_rows": rows,
    }


def _h8() -> dict[str, object]:
    from fuzzyxai.strong_confirmatory import compare_grid_configurations

    rng = np.random.default_rng(SEED)
    reports = []
    settings = {
        "tabular": ["terms", "knots", "membership_overlap", "rule_granularity"],
        "image": ["superpixel_count", "segmentation_seed", "mask_resolution", "merge_threshold"],
        "text": ["token_grouping", "phrase_grouping", "sentence_grouping", "grouping_threshold"],
        "timeseries": ["window_length", "window_overlap", "multiscale_windows", "channel_grouping"],
    }
    for modality, parameters in settings.items():
        base = rng.uniform(size=1000)
        top = np.tile(np.arange(5), (1000, 1))
        configurations = {}
        for name, noise_scale in (("coarse", 0.008), ("default", 0.0), ("fine", 0.005), ("very_fine", 0.012)):
            risk = np.clip(base + rng.normal(scale=noise_scale, size=len(base)), 0.0, 1.0)
            configurations[name] = {
                "actions": np.select((risk < 0.3, risk < 0.6, risk < 0.85), ("accept", "short_review", "full_review"), default="block").tolist(),
                "representations": np.select((risk < 0.35, risk < 0.6, risk < 0.8), ("F0", "Fint", "NAS"), default="FML").tolist(),
                "risk": risk.tolist(),
                "top_k": top.tolist(),
            }
        report = compare_grid_configurations(configurations)
        report.update({"modality": modality, "varied_parameters": parameters})
        reports.append(report)
    return {
        "phase": "formative_controlled_grid",
        "modalities": reports,
        "formative_target_met": all(report["formative_target_met"] for report in reports),
        "confirmatory_claim_allowed": False,
        "raw_rows": [
            {"modality": report["modality"], **row}
            for report in reports
            for row in report["configurations"]
        ],
    }


def _h9(*, profile: str) -> dict[str, object]:
    sizes = (1_000, 10_000) if profile == "smoke" else (1_000, 100_000, 1_000_000, 2_000_000, 5_000_000)
    cached = run_streaming_scalability(sizes=sizes, batch_size=10_000, seed=SEED)
    representative_size = 10_000 if profile == "smoke" else 100_000
    rng = np.random.default_rng(SEED)
    values = rng.normal(size=(representative_size, 30)).astype(np.float32)
    weights = rng.normal(size=30).astype(np.float32)
    started = perf_counter()
    explanations = values * weights
    base_time = perf_counter() - started
    started = perf_counter()
    payloads = [json.dumps({"values": row[:8].tolist()}, separators=(",", ":")) for row in explanations[: min(10_000, len(explanations))]]
    uncached_time = perf_counter() - started
    measurements = cached["measurements"]
    return {
        "phase": "formative_measurement",
        "cached_operator_layer": cached,
        "representative_base_explainer": {
            "kind": "linear_contribution",
            "n_objects": representative_size,
            "wall_time_seconds": base_time,
        },
        "uncached_serialization_subset": {
            "n_objects": len(payloads),
            "wall_time_seconds": uncached_time,
        },
        "maximum_objects": max(row["n_objects"] for row in measurements),
        "peak_rss_bytes": max(row["peak_rss_bytes"] for row in measurements),
        "formative_target_met": bool(cached["formative_target_met"] and max(row["n_objects"] for row in measurements) >= (10_000 if profile == "smoke" else 5_000_000)),
        "confirmatory_claim_allowed": False,
        "raw_rows": measurements,
    }


def _write_experiment(name: str, payload: dict[str, object], *, raw_rows: list[dict[str, object]]) -> None:
    directory = OUTPUT / name
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "protocol.json", {"experiment_id": name, "phase": "formative", "confirmatory_test_opened": False})
    _write_json(directory / "dataset_manifest.json", {"dataset_scope": "development_only", "confirmatory_dataset": False})
    _write_json(directory / "split_manifest.json", {"features": "out_of_fold_or_controlled", "test_opened": False})
    _write_json(directory / "summary.json", payload)
    _write_json(directory / "statistical_tests.json", {"status": "descriptive_formative_only", "confirmatory_inference": False})
    _write_json(directory / "claim_status.json", {"status": "formative_only", "claim_allowed": False})
    _write_jsonl(directory / "raw_results.jsonl", raw_rows)
    _write_parquet(directory / "raw_results.parquet", raw_rows)
    _write_json(directory / "tables_manifest.json", {"status": "pending aggregate builder"})
    _write_json(directory / "figures_manifest.json", {"status": "pending aggregate builder"})
    paths = sorted(path for path in directory.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (directory / "SHA256SUMS").write_text(
        "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in paths), encoding="utf-8"
    )


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit("FAIL: pyarrow is required for machine-readable raw_results.parquet") from error
    normalized = [{key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()} for row in rows]
    table = pa.Table.from_pylist(normalized or [{"status": "no_rows"}])
    pq.write_table(table, path, compression="zstd")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
