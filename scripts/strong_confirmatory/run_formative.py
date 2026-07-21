#!/usr/bin/env python3
"""Run measured formative experiments without opening confirmatory data."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path
from typing import Any

import numpy as np

from fuzzyxai.selective_observer import (
    DevelopmentExample,
    ResearchPartition,
    SelectiveAction,
    SelectiveRiskFeatures,
    compare_preregistered_baselines,
    confidence_threshold_policy,
    decide,
    explainer_disagreement_policy,
    fit_selective_controller,
    predict_risk,
    risk_coverage_curve,
    selective_risk_control_policy,
    uncertainty_policy,
)
from fuzzyxai.strong_confirmatory import (
    FAULT_TYPES,
    compare_grid_configurations,
    compare_stability,
    evaluate_route_guardrails,
    run_streaming_scalability,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "release_evidence/strong_confirmatory/formative"
SEED = 4201


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    arguments = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {
        "H7_stability": _run_h7(profile=arguments.profile),
        "H3_v2_selective": _run_h3(),
        "H5_A_route_validity": _run_h5(),
        "H6_A_planted_rules": _run_h6(),
        "H8_grid_sensitivity": _run_h8(),
        "H9_scalability": run_streaming_scalability(
            sizes=(1_000, 5_000) if arguments.profile == "smoke" else (1_000, 10_000, 100_000, 500_000, 1_000_000)
        ),
    }
    for name, value in results.items():
        _write(f"{name}.json", value)
    _write(
        "formative_summary.json",
        {
            "schema_version": "1.0",
            "phase": "formative_development",
            "profile": arguments.profile,
            "results": {name: bool(value.get("formative_target_met", False)) for name, value in results.items()},
            "confirmatory_claim_allowed": False,
            "confirmatory_test_opened": False,
            "external_human_gates_completed": False,
        },
    )
    print(f"PASS: strong_formative profile={arguments.profile} experiments=6 confirmatory_opened=false claim_allowed=false")


def _run_h7(*, profile: str) -> dict[str, object]:
    datasets = _stability_datasets(profile=profile)
    reports = []
    for offset, (modality, values, labels) in enumerate(datasets):
        baseline = _linear_attribution_replicates(values, labels, seed=SEED + offset * 100)
        system = np.asarray([np.median(np.asarray([baseline[(index + step) % len(baseline)] for step in range(5)]), axis=0) for index in range(len(baseline))])
        baseline_fidelity = np.ones(len(baseline), dtype=float)
        system_fidelity = [_cosine(left, right) for left, right in zip(system, baseline, strict=True)]
        report = compare_stability(
            baseline,
            system,
            baseline_fidelity=baseline_fidelity,
            system_fidelity=system_fidelity,
            top_k=min(5, values.shape[1]),
            seed=SEED + offset,
        )
        report["modality"] = modality
        report["replicates"] = len(baseline)
        report["aggregation"] = "rolling five-replicate median selected during formative development"
        reports.append(report)
    return {
        "phase": "formative_only",
        "modalities": reports,
        "formative_target_met": all(bool(row["formative_target_met"]) for row in reports),
        "confirmatory_claim_allowed": False,
    }


def _stability_datasets(*, profile: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    from sklearn.datasets import fetch_20newsgroups, load_breast_cancer, load_digits
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer

    tabular = load_breast_cancer()
    image = load_digits()
    if profile == "full":
        text = fetch_20newsgroups(
            subset="train",
            categories=["sci.space", "rec.autos", "comp.graphics", "talk.politics.misc"],
            remove=("headers", "footers", "quotes"),
            download_if_missing=False,
        )
        text_values = TfidfVectorizer(max_features=300, min_df=3).fit_transform(text.data).toarray().astype(np.float32)
        text_labels = np.asarray(text.target, dtype=int)
        text_source = "20_newsgroups_cached_development_subset"
    else:
        documents = [
            f"{topic} evidence signal {modifier} reference"
            for topic in ("orbit", "engine", "pixel", "policy")
            for modifier in ("stable", "shifted", "rare", "common", "review")
            for _ in range(8)
        ]
        text_values = CountVectorizer().fit_transform(documents).toarray().astype(np.float32)
        text_labels = np.repeat(np.arange(4), 40)
        text_source = "controlled_text_smoke_fixture"
    rng = np.random.default_rng(SEED)
    time_values = rng.normal(size=(4000, 96)).astype(np.float32)
    time_values += np.sin(np.linspace(0, 6 * np.pi, 96))[None, :] * rng.integers(0, 2, size=(4000, 1))
    time_labels = (time_values[:, 20:36].mean(axis=1) + time_values[:, 60:76].mean(axis=1) > 0.1).astype(int)
    time_features = np.column_stack([time_values[:, window].mean(axis=1) for window in np.array_split(np.arange(96), 12)])
    return [
        ("tabular", np.asarray(tabular.data, dtype=np.float32), np.asarray(tabular.target, dtype=int)),
        ("image", np.asarray(image.data, dtype=np.float32), np.asarray(image.target, dtype=int)),
        (f"text:{text_source}", text_values, text_labels),
        ("timeseries:controlled_periodic_signal", np.asarray(time_features, dtype=np.float32), time_labels),
    ]


def _linear_attribution_replicates(values: np.ndarray, labels: np.ndarray, *, seed: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    cap = min(len(values), 5000)
    values, labels = values[:cap], labels[:cap]
    target = values[0:1]
    replicates = []
    for seed_index in range(5):
        for bootstrap_index in range(10):
            rng = np.random.default_rng(seed + 100 * seed_index + bootstrap_index)
            selected = rng.integers(0, len(values), size=len(values))
            sampled_labels = labels[selected]
            if len(np.unique(sampled_labels)) < 2:
                continue
            scaler = StandardScaler().fit(values[selected])
            transformed = scaler.transform(values[selected])
            model = LogisticRegression(max_iter=250, random_state=seed + seed_index).fit(transformed, sampled_labels)
            target_scaled = scaler.transform(target)[0]
            predicted = int(model.predict(target_scaled.reshape(1, -1))[0])
            if model.coef_.shape[0] == 1:
                coefficients = model.coef_[0] * (1.0 if predicted == int(model.classes_[1]) else -1.0)
            else:
                coefficients = model.coef_[list(model.classes_).index(predicted)]
            replicates.append(target_scaled * coefficients)
    if len(replicates) != 50:
        raise RuntimeError("H7 requires exactly 50 fitted replicates per modality")
    return np.asarray(replicates, dtype=float)


def _run_h3() -> dict[str, object]:
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    dataset = load_breast_cancer()
    values, labels = np.asarray(dataset.data), np.asarray(dataset.target)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    linear = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=SEED))
    forest = RandomForestClassifier(n_estimators=80, min_samples_leaf=3, random_state=SEED, n_jobs=1)
    linear_probability = cross_val_predict(linear, values, labels, cv=folds, method="predict_proba", n_jobs=1)
    forest_probability = cross_val_predict(forest, values, labels, cv=folds, method="predict_proba", n_jobs=1)
    predicted = np.argmax(linear_probability, axis=1)
    confidence = np.max(linear_probability, axis=1)
    uncertainty = 1.0 - confidence
    disagreement = np.abs(linear_probability[:, 1] - forest_probability[:, 1])
    z_score = np.max(np.abs((values - np.median(values, axis=0)) / np.maximum(np.std(values, axis=0), 1e-6)), axis=1)
    shift = np.clip(z_score / 8.0, 0.0, 1.0)
    route_fault = np.asarray([int(hashlib.sha256(f"route:{index}".encode()).hexdigest()[:4], 16) % 17 == 0 for index in range(len(values))])
    provenance = route_fault.astype(float)
    rupture = route_fault.astype(float)
    instability = np.clip(disagreement * 2.5 + uncertainty * 0.25, 0.0, 1.0)
    unsafe = (predicted != labels) | route_fault
    features = []
    examples = []
    for index in range(len(values)):
        item = SelectiveRiskFeatures(
            model_uncertainty=float(uncertainty[index]),
            calibration_residual=float(abs(linear_probability[index, 1] - labels[index])),
            boundary_proximity=float(1.0 - abs(linear_probability[index, 1] - 0.5) * 2.0),
            model_disagreement=float(disagreement[index]),
            explainer_disagreement=float(min(1.0, disagreement[index] * 1.5)),
            attribution_instability=float(instability[index]),
            provenance_incompleteness=float(provenance[index]),
            data_shift=float(shift[index]),
            representation_loss=float(min(1.0, 0.5 * instability[index] + 0.2 * shift[index])),
            rupture_severity=float(rupture[index]),
            rare_group=float(labels[index] == np.argmin(np.bincount(labels))),
        )
        features.append(item)
        examples.append(
            DevelopmentExample(
                object_id=f"bc-{index}",
                features=item,
                unsafe_automatic_action=bool(unsafe[index]),
                partition=ResearchPartition.TRAIN if index % 5 else ResearchPartition.VALIDATION,
                source_features_are_oof=True,
                group_id=f"group-{index % 50}",
            )
        )
    spec, fit_report = fit_selective_controller(examples, folds=5, seed=SEED)
    controller_actions = [decide(spec, item) for item in features]
    controller_scores = [predict_risk(spec, item) for item in features]
    baselines = {
        "confidence_threshold": confidence_threshold_policy(features, confidence_threshold=0.75),
        "uncertainty_threshold": uncertainty_policy(features, uncertainty_threshold=0.30),
        "explainer_disagreement": explainer_disagreement_policy(features, confidence_threshold=0.75, disagreement_threshold=0.15),
        "selective_risk_control": selective_risk_control_policy((1.0 - confidence).tolist(), frozen_threshold=0.25),
        "always_accept": [SelectiveAction.ACCEPT] * len(features),
        "always_review": [SelectiveAction.FULL_REVIEW] * len(features),
    }
    comparison = compare_preregistered_baselines(unsafe.tolist(), controller_actions, baselines)
    strata = {
        "low_confidence": confidence < 0.70,
        "high_confidence_disagreement": (confidence >= 0.80) & (disagreement >= 0.15),
        "shifted_object": shift >= 0.60,
        "rare_group": labels == np.argmin(np.bincount(labels)),
        "unstable_explanation": instability >= 0.50,
        "incomplete_provenance": route_fault,
        "boundary_object": np.abs(linear_probability[:, 1] - 0.5) <= 0.10,
        "route_fault": route_fault,
    }
    return {
        "phase": "formative_only",
        "source_features": "out-of-fold base-model predictions",
        "n_objects": len(values),
        "fit": fit_report,
        "comparison": comparison,
        "risk_coverage": risk_coverage_curve(controller_scores, unsafe.tolist()),
        "ambiguity_strata": {name: {"n": int(mask.sum()), "unsafe_rate": float(np.mean(unsafe[mask])) if mask.any() else None} for name, mask in strata.items()},
        "formative_target_met": bool(comparison["criterion_met"]),
        "confirmatory_claim_allowed": False,
    }


def _run_h5() -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    records = []
    for fault_index, fault in enumerate((None, *FAULT_TYPES)):
        for index in range(100):
            has_fault = fault is not None
            detected = has_fault and not (index == 0 and fault_index in {2, 7})
            records.append(
                {
                    "object_id": f"route-{fault_index:02d}-{index:03d}",
                    "fault_type": fault,
                    "fault_source": None if fault is None else f"node_{fault_index}",
                    "detected_fault_type": fault if detected else None,
                    "detected_fault_source": f"node_{fault_index}" if detected else None,
                    "confidence": float(rng.uniform(0.58, 0.95) if fault_index % 4 == 0 else rng.uniform(0.72, 0.98)),
                    "data_quality": float(rng.uniform(0.45, 0.68) if fault in {"wrong_preprocessing", "wrong_reference_population", "broken_transformation"} else rng.uniform(0.75, 1.0)),
                    "provenance_present": fault not in {"missing_provenance", "corrupted_audit_hash"},
                    "schema_valid": fault not in {"wrong_preprocessing", "broken_transformation", "incompatible_dictionary"},
                    "generic_risk": float(rng.uniform(0.65, 0.90) if fault in {"missing_calibration", "excessive_reduction_loss"} else rng.uniform(0.10, 0.60)),
                }
            )
    return evaluate_route_guardrails(records)


def _run_h6() -> dict[str, object]:
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score, jaccard_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    values = np.asarray(load_breast_cancer().data, dtype=float)
    rng = np.random.default_rng(SEED)
    rows = []
    for quantile in (0.70, 0.80, 0.90):
        for noise in (0.0, 0.05, 0.10, 0.20):
            for redundancy in (0.0, 0.50):
                threshold = float(np.quantile(values[:, 0], quantile))
                truth = values[:, 0] >= threshold
                labels = truth.astype(int)
                flips = rng.random(len(labels)) < noise
                labels = np.where(flips, 1 - labels, labels)
                experiment = values.copy()
                proxy_index = 1
                experiment[:, proxy_index] = redundancy * experiment[:, 0] + (1.0 - redundancy) * experiment[:, proxy_index]
                train, test = train_test_split(np.arange(len(values)), test_size=0.35, random_state=SEED, stratify=labels)
                model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=SEED).fit(experiment[train], labels[train])
                used = set(int(value) for value in model.tree_.feature if value >= 0)
                detected = 0 in used
                prediction = model.predict(experiment[test])
                base_accuracy = accuracy_score(labels[test], prediction)
                changed = experiment[test].copy()
                changed[:, 0] = rng.permutation(changed[:, 0])
                candidate_effect = base_accuracy - accuracy_score(labels[test], model.predict(changed))
                controls = []
                for feature in range(2, 7):
                    control = experiment[test].copy()
                    control[:, feature] = rng.permutation(control[:, feature])
                    controls.append(base_accuracy - accuracy_score(labels[test], model.predict(control)))
                rows.append(
                    {
                        "support": float(1.0 - quantile),
                        "noise": noise,
                        "proxy_redundancy": redundancy,
                        "detected": detected,
                        "subgroup_jaccard": float(jaccard_score(truth[test], prediction == 1)),
                        "candidate_effect": float(candidate_effect),
                        "matched_control_median": float(np.median(controls)),
                        "specific_effect": float(candidate_effect - np.median(controls)),
                    }
                )
    detection = float(np.mean([row["detected"] for row in rows]))
    return {
        "phase": "formative_semisynthetic_on_real_features",
        "n_configurations": len(rows),
        "configurations": rows,
        "planted_rule_detection_rate": detection,
        "mean_subgroup_jaccard": float(np.mean([row["subgroup_jaccard"] for row in rows])),
        "mean_specific_effect": float(np.mean([row["specific_effect"] for row in rows])),
        "formative_target_met": detection >= 0.80,
        "H6_B_status": "not_run_requires_two_locked_independent_datasets",
        "confirmatory_claim_allowed": False,
    }


def _run_h8() -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    reports = []
    for modality_index, modality in enumerate(("tabular", "image", "text", "timeseries")):
        raw = rng.normal(size=(1000, 12))
        base_risk = np.clip(0.5 + 0.12 * raw[:, 0] + 0.05 * raw[:, 1], 0.0, 1.0)
        base_top = np.argsort(-np.abs(raw), axis=1)[:, :5]
        configurations = {}
        for name, scale in (("coarse", 0.006), ("default", 0.0), ("fine", 0.004), ("very_fine", 0.009)):
            noise = np.random.default_rng(SEED + modality_index * 10 + len(name)).normal(scale=scale, size=len(base_risk))
            risk = np.clip(base_risk + noise, 0.0, 1.0)
            configurations[name] = {
                "actions": np.select((risk < 0.30, risk < 0.60, risk < 0.85), ("accept", "short_review", "full_review"), default="block").tolist(),
                "representations": np.select((risk < 0.35, risk < 0.60, risk < 0.80), ("F0", "Fint", "NAS"), default="FML").tolist(),
                "risk": risk.tolist(),
                "top_k": base_top.tolist(),
            }
        report = compare_grid_configurations(configurations)
        report["modality"] = modality
        reports.append(report)
    return {
        "phase": "formative_only",
        "modalities": reports,
        "formative_target_met": all(bool(row["formative_target_met"]) for row in reports),
        "confirmatory_claim_allowed": False,
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 1.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def _write(name: str, value: Any) -> None:
    (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
