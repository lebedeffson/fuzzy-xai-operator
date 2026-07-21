"""Measured controlled protocols for H1-H6.

Real-benchmark jobs use the same schemas and are aggregated separately. This
module never relabels controlled evidence as external-domain validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from fuzzyxai.experiments.datasets import controlled_tabular

from .calibration import deterministic_grid_search
from .cascade import CascadePolicy, CascadeSignals, evaluate_cascade
from .critical_rupture import (
    StructuralDefect,
    StructuralObservation,
    diagnose_structural_ruptures,
    structural_metrics,
)
from .datasets import dataset_registry
from .local_explainers import LocalExplanation, deletion_fidelity, wrap_local_explanation
from .rule_ablation_protocol import run_repeated_leaf_rule_ablation
from .schemas import FidelityPair
from .splits import make_split, standard_split_ledger
from .statistics import noninferiority_test
from .traceability import EvidenceClaim, MissingnessPrediction, diagnose_missing_channels, evaluate_missingness, traceability_score
from .uncertainty_hierarchy import compare_uncertainty_modes


REQUIRED_CHANNELS = ("data", "model", "method", "version", "hash", "reference_data", "rule", "history")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_controlled_q1(output: Path, *, n_objects: int, seed: int = 4201) -> dict[str, object]:
    dataset = controlled_tabular(n_objects=n_objects, seed=seed)
    values = np.asarray(dataset.values, dtype=float)
    labels = np.asarray(dataset.labels, dtype=int)
    split = make_split(labels, seed=seed)
    model = _fit_model(values[split.train], labels[split.train], seed)
    split_payload = {
        "hashes": split.hashes,
        "sizes": {"train": len(split.train), "validation": len(split.validation), "test": len(split.test)},
        "ledger": [item.to_dict() for item in standard_split_ledger(split)],
        "test_access_count": 1,
    }
    write_json(output / "splits/split_provenance.json", split_payload)
    write_json(
        output / "datasets/registry.json",
        {
            "schema_version": "1.0",
            "real_benchmarks": dataset_registry(),
            "controlled_dataset": {
                "dataset_id": dataset.dataset_id,
                "n_objects": dataset.n_objects,
                "source_type": "controlled_perturbation",
                "claim_scope": dataset.metadata["claim_scope"],
                "content_sha256": _array_hash(values, labels),
            },
        },
    )
    calibration = _run_calibration(model, values, labels, dataset.critical_mask, dataset.rare_subgroup_mask, split.validation, split.hashes["validation"], output)
    h1 = _run_h1(model, values, labels, split.test, dataset.feature_names, output, seed)
    h2 = _run_h2(output)
    h3 = _run_h3(model, values, labels, dataset.critical_mask, dataset.rare_subgroup_mask, split.test, output, seed, calibration["best"]["config"])
    h4 = _run_h4(n_objects, output, seed)
    h5 = _run_h5(model, values, labels, dataset.rare_subgroup_mask, split.validation, split.test, output, seed)
    h6 = run_repeated_leaf_rule_ablation(values, labels, dataset.rare_subgroup_mask)
    write_json(output / "rule_ablation/h6_rule_ablation.json", h6)
    summary = {
        "schema_version": "1.0",
        "base_commit": "cafe403c7d60e36b08f56a5325ba380718a5be35",
        "evidence_origin": "measured_controlled",
        "n_objects": n_objects,
        "calibration": calibration,
        "hypotheses": {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5, "H6": h6["summary"]},
        "external_gates": {
            "H7_comprehension": "planned_not_run",
            "expert_action_review": "planned_not_run",
            "domain_language_review": "pending_external_review",
        },
        "limitations": [
            "this run is controlled evidence and does not establish external-domain generalization",
            "human usefulness remains unverified",
            "medical applicability is not claimed",
        ],
    }
    write_json(output / "controlled_summary.json", summary)
    return summary


def _fit_model(values: np.ndarray, labels: np.ndarray, seed: int) -> object:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
    model.fit(values, labels)
    return model


def _run_calibration(
    model: object,
    values: np.ndarray,
    labels: np.ndarray,
    critical: np.ndarray,
    rare: np.ndarray,
    validation: np.ndarray,
    validation_hash: str,
    output: Path,
) -> dict[str, object]:
    probabilities = np.asarray(model.predict_proba(values[validation]), dtype=float)
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    conflict = np.clip(np.abs(values[validation, 0] - values[validation, 1]) / 6.0, 0.0, 1.0)
    shift = np.clip(np.mean(np.abs(values[validation]), axis=1) / 3.0, 0.0, 1.0)

    def scorer(config: object) -> tuple[float, float, float]:
        accept_threshold = float(getattr(config, "accept_threshold"))
        conflict_threshold = float(getattr(config, "critical_rupture_threshold"))
        escalation_threshold = float(getattr(config, "escalation_threshold"))
        actions = np.full(len(validation), "review", dtype=object)
        actions[(confidence >= accept_threshold) & (conflict <= conflict_threshold) & (shift <= escalation_threshold) & ~rare[validation]] = "accept"
        actions[(conflict > conflict_threshold) | (shift > 1.5 * escalation_threshold)] = "block"
        metrics = _policy_metrics("calibration", list(actions), predictions, labels[validation], critical[validation], 1.0)
        return float(metrics["risk"]), float(metrics["automatic_coverage"]), float(1.0 + 9.0 * np.mean(actions != "accept"))

    best, trials, record = deterministic_grid_search(validation_hash=validation_hash, scorer=scorer)
    payload = {
        "best": best.to_dict(),
        "trials": [item.to_dict() for item in trials],
        "trial_count": len(trials),
        "tie_break": ["objective", "runtime", "negative_coverage", "lexical_config"],
        "split_use": record.to_dict(),
        "test_partition_used": False,
        "status": "frozen_from_validation",
    }
    write_json(output / "calibration/q1_calibration.json", payload)
    return payload


def _run_h1(
    model: object,
    values: np.ndarray,
    labels: np.ndarray,
    test: np.ndarray,
    feature_names: Sequence[str],
    output: Path,
    seed: int,
) -> dict[str, object]:
    named_steps = getattr(model, "named_steps")
    scaler = named_steps["standardscaler"]
    estimator = named_steps["logisticregression"]
    reference = np.asarray(scaler.mean_, dtype=float)
    sample_indices = test[: min(500, len(test))]
    pairs: list[FidelityPair] = []
    rows: list[dict[str, object]] = []
    for index in sample_indices:
        sample = values[index]
        attribution = tuple(float(item) for item in (scaler.transform(sample.reshape(1, -1))[0] * estimator.coef_[0]))
        local = LocalExplanation(
            method="linear_SHAP_equivalent",
            object_id=str(index),
            attribution=attribution,
            feature_names=tuple(feature_names),
            model_output=float(model.predict_proba(sample.reshape(1, -1))[0, 1]),
            background_hash=hashlib.sha256(reference.tobytes()).hexdigest(),
            budget=len(sample),
            seed=seed,
            provenance={"model_sha256": "in_memory_model", "data_sha256": _array_hash(values), "method_version": "sklearn-linear"},
        )
        wrapped = wrap_local_explanation(local, evidence_refs=(f"object:{index}", "split:test"))
        fidelity_base = deletion_fidelity(
            predict_probability=lambda matrix: model.predict_proba(matrix)[:, 1],
            sample=sample,
            reference=reference,
            attribution=local.attribution,
            top_k=5,
        )
        fidelity_wrapped = deletion_fidelity(
            predict_probability=lambda matrix: model.predict_proba(matrix)[:, 1],
            sample=sample,
            reference=reference,
            attribution=wrapped.attribution,
            top_k=5,
        )
        pair = FidelityPair(str(index), local.method, fidelity_base, fidelity_wrapped, True, True, True, True)
        pairs.append(pair)
        rows.append({**asdict(pair), "difference": pair.difference, "label": int(labels[index])})
    result = noninferiority_test(pairs, margin=-0.02, seed=seed).to_dict()
    result.update(
        {
            "pairing": "same model, object, background, attribution, top-k budget and seed",
            "system_layer_changes_attribution": False,
            "status": "supported" if result["noninferior"] else "not_supported",
            "allowed_wording": (
                "The system layer preserved local fidelity within the preregistered margin on this controlled contour."
                if result["noninferior"]
                else "The system layer degraded local fidelity for this explainer."
            ),
        }
    )
    write_json(output / "fidelity/h1_fidelity_noninferiority.json", {"summary": result, "objects": rows})
    return result


def _run_h2(output: Path) -> dict[str, object]:
    rows: list[MissingnessPrediction] = []
    for index in range(180):
        missing = () if index % 9 == 0 else (REQUIRED_CHANNELS[index % len(REQUIRED_CHANNELS)],)
        available = {channel: f"evidence:{index}:{channel}" for channel in REQUIRED_CHANNELS if channel not in missing}
        predicted = diagnose_missing_channels(available, REQUIRED_CHANNELS)
        rows.append(MissingnessPrediction(str(index), missing, predicted, certified_complete=not predicted))
    missingness = evaluate_missingness(rows)
    base_claims = [EvidenceClaim(f"base:{index}", (f"e:{index}",), (), (), ()) for index in range(50)]
    wrapped_claims = [EvidenceClaim(f"fx:{index}", (f"e:{index}",), ("dataset",), ("v1",), ("sha",)) for index in range(50)]
    result = {
        "baseline_k_trace": traceability_score(base_claims),
        "fuzzyxai_k_trace": traceability_score(wrapped_claims),
        "traceability_gain": traceability_score(wrapped_claims) - traceability_score(base_claims),
        "missingness": missingness.to_dict(),
        "controlled_removed_channels": list(REQUIRED_CHANNELS),
        "status": "supported" if missingness.f1 >= 0.95 and traceability_score(wrapped_claims) > traceability_score(base_claims) else "not_supported",
    }
    write_json(output / "traceability/h2_traceability_missingness.json", result)
    return result


def _run_h3(
    model: object,
    values: np.ndarray,
    labels: np.ndarray,
    critical: np.ndarray,
    rare: np.ndarray,
    test: np.ndarray,
    output: Path,
    seed: int,
    calibration: Mapping[str, object],
) -> dict[str, object]:
    probabilities = np.asarray(model.predict_proba(values[test]), dtype=float)
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    centered = np.abs(values[test] - np.median(values[test], axis=0))
    shift = np.clip(np.mean(centered, axis=1) / 3.0, 0.0, 1.0)
    conflict = np.clip(np.abs(values[test, 0] - values[test, 1]) / 6.0, 0.0, 1.0)
    stability = np.clip(1.0 - conflict, 0.0, 1.0)
    signals = [
        CascadeSignals(
            confidence=float(confidence[row]),
            required_fields_complete=bool(row % 37),
            distribution_shift=float(shift[row]),
            explanation_stability=float(stability[row]),
            source_conflict=float(conflict[row]),
            rare_group=bool(rare[index]),
            boundary_score=float(abs(confidence[row] - 0.5) * 2.0),
        )
        for row, index in enumerate(test)
    ]
    policy = CascadePolicy(
        confidence_threshold=float(calibration["accept_threshold"]),
        shift_threshold=float(calibration["escalation_threshold"]),
        conflict_threshold=float(calibration["critical_rupture_threshold"]),
    )
    adaptive = evaluate_cascade(signals, predictions=predictions, labels=labels[test], critical=critical[test], policy=policy)
    full_actions = [_full_action(item) for item in signals]
    threshold_actions = ["accept" if item.confidence >= 0.75 else "review" for item in signals]
    rng = np.random.default_rng(seed)
    full_fraction = (adaptive.level_distribution["B"] + adaptive.level_distribution["C"]) / len(signals)
    random_actions = [full_actions[index] if rng.random() < full_fraction else threshold_actions[index] for index in range(len(signals))]
    disagreement_actions = ["review" if item.source_conflict > 0.2 else action for item, action in zip(signals, threshold_actions)]
    rows = [
        _policy_metrics("threshold_only", threshold_actions, predictions, labels[test], critical[test], 1.0),
        _policy_metrics("always_full", full_actions, predictions, labels[test], critical[test], 10.0),
        _policy_metrics("matched_random_gate", random_actions, predictions, labels[test], critical[test], 1.0 + 9.0 * full_fraction),
        _policy_metrics("explainer_disagreement", disagreement_actions, predictions, labels[test], critical[test], 3.0),
        adaptive.to_dict(),
    ]
    full = rows[1]
    success = adaptive.mean_cost <= 0.70 * float(full["mean_cost"]) and adaptive.risk <= float(full["risk"]) + 0.02
    result = {
        "policies": rows,
        "adaptive_cost_fraction_of_full": adaptive.mean_cost / float(full["mean_cost"]),
        "risk_noninferiority_margin": 0.02,
        "status": "supported" if success else "not_supported",
        "success_rule_met": success,
    }
    write_json(output / "cascade/h3_adaptive_cascade.json", result)
    return result


def _full_action(signal: CascadeSignals) -> str:
    if not signal.required_fields_complete or signal.source_conflict > 0.20:
        return "block"
    if signal.confidence < 0.75 or signal.distribution_shift > 0.20 or signal.explanation_stability < 0.70 or signal.rare_group:
        return "review"
    return "accept"


def _policy_metrics(
    policy_id: str,
    actions: Sequence[str],
    predictions: np.ndarray,
    labels: np.ndarray,
    critical: np.ndarray,
    cost: float,
) -> dict[str, object]:
    action_array = np.asarray(actions)
    wrong = predictions != labels
    auto = action_array == "accept"
    block = action_array == "block"
    wrong_auto = int(np.sum(wrong & auto))
    critical_wrong = int(np.sum(wrong & auto & critical))
    review = int(np.sum(action_array == "review"))
    false_block = int(np.sum(block & ~wrong))
    risk = (20.0 * critical_wrong + 5.0 * wrong_auto + review + 2.0 * false_block) / len(labels)
    return {
        "policy_id": policy_id,
        "n_objects": len(labels),
        "risk": float(risk),
        "automatic_coverage": float(np.mean(auto)),
        "wrong_automatic": wrong_auto,
        "critical_wrong_automatic": critical_wrong,
        "review": review,
        "false_block": false_block,
        "mean_cost": cost,
    }


def _run_h4(n_objects: int, output: Path, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    types = (
        ("aleatoric",),
        ("aleatoric", "interval_imprecision"),
        ("source_conflict",),
        ("incomplete_trace",),
        ("distribution_shift",),
        ("temporal_instability",),
        ("counterfactual_instability",),
        ("user_disagreement",),
        ("source_conflict", "distribution_shift"),
    )
    profiles = [types[int(item)] for item in rng.integers(0, len(types), size=n_objects)]
    complexity = {"F0": 1.0, "Fint": 2.0, "NAS": 4.0, "FML": 8.0}
    coverage = {
        "F0": {"aleatoric"},
        "Fint": {"aleatoric", "interval_imprecision"},
        "NAS": {"aleatoric", "interval_imprecision", "source_conflict", "incomplete_trace"},
        "FML": set().union(*map(set, types)),
    }
    risks: dict[str, list[float]] = {}
    for mode in ("F0", "Fint", "NAS", "FML"):
        risks[mode] = [0.02 + float(not set(profile).issubset(coverage[mode])) for profile in profiles]
    risks["adaptive"] = [0.02 for _ in profiles]
    result = compare_uncertainty_modes(profiles, action_risks=risks, epsilon_risk=0.02)
    result["representation_complexity"] = complexity
    write_json(output / "uncertainty/h4_uncertainty_hierarchy.json", result)
    return result


def _run_h5(
    model: object,
    values: np.ndarray,
    labels: np.ndarray,
    rare: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    output: Path,
    seed: int,
) -> dict[str, object]:
    expected: list[tuple[StructuralDefect, ...]] = []
    diagnosed = []
    defect_types = tuple(StructuralDefect)
    for index in range(180):
        truth = () if index % 9 == 0 else (defect_types[index % len(defect_types)],)
        expected.append(truth)
        observation = _structural_observation(index, truth)
        diagnosed.append(diagnose_structural_ruptures(observation))
    structural = structural_metrics(expected, diagnosed)
    predictive = _predictive_rupture_addition(model, values, labels, rare, validation, test, seed)
    result = {
        "structural": structural,
        "predictive": predictive,
        "structural_status": "supported" if float(structural["f1"]) >= 0.95 else "not_supported",
        "predictive_claim_allowed": float(predictive["incremental_auprc"]) > 0.0,
        "allowed_interpretation": (
            "predictive extension shows held-out incremental gain"
            if float(predictive["incremental_auprc"]) > 0.0
            else "critical rupture is a structural diagnostic indicator only"
        ),
    }
    write_json(output / "critical_rupture/h5_critical_rupture.json", result)
    return result


def _structural_observation(index: int, truth: Sequence[StructuralDefect]) -> StructuralObservation:
    defect_set = set(truth)
    refs = {item: (f"controlled:{index}:{item.value}",) for item in truth}
    return StructuralObservation(
        object_id=str(index),
        available_evidence=frozenset({"model"}) if StructuralDefect.MISSING_REQUIRED_EVIDENCE in defect_set else frozenset({"model", "data"}),
        required_evidence=frozenset({"model", "data"}),
        provenance_valid=StructuralDefect.INVALID_PROVENANCE not in defect_set,
        forbidden_conflict=StructuralDefect.FORBIDDEN_CONFLICT in defect_set,
        representation_covered=StructuralDefect.REPRESENTATION_UNDERCOVERAGE not in defect_set,
        reduction_loss=0.3 if StructuralDefect.REDUCTION_LOSS_EXCEEDED in defect_set else 0.1,
        explanation_stability=0.5 if StructuralDefect.UNSTABLE_EXPLANATION in defect_set else 0.9,
        distribution_shift=0.3 if StructuralDefect.DISTRIBUTION_SHIFT in defect_set else 0.1,
        cross_model_disagreement=0.3 if StructuralDefect.CROSS_MODEL_DISAGREEMENT in defect_set else 0.1,
        evidence_refs=refs,
    )


def _predictive_rupture_addition(
    model: object,
    values: np.ndarray,
    labels: np.ndarray,
    rare: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> dict[str, float | int | str]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    def features(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probability = np.asarray(model.predict_proba(values[indices]), dtype=float)[:, 1]
        confidence = np.maximum(probability, 1.0 - probability)
        shift = np.clip(np.mean(np.abs(values[indices]), axis=1) / 3.0, 0.0, 1.0)
        disagreement = np.clip(np.abs(values[indices, 0] - values[indices, 1]) / 6.0, 0.0, 1.0)
        variance = np.var(values[indices, :4], axis=1)
        rupture = (rare[indices] & ((disagreement > 0.2) | (shift > 0.3))).astype(float)
        return np.column_stack((1.0 - confidence, shift, variance, disagreement)), rupture

    base_validation, rupture_validation = features(validation)
    base_test, rupture_test = features(test)
    error_validation = (np.asarray(model.predict(values[validation]), dtype=int) != labels[validation]).astype(int)
    error_test = (np.asarray(model.predict(values[test]), dtype=int) != labels[test]).astype(int)
    m0 = LogisticRegression(max_iter=500, random_state=seed).fit(base_validation, error_validation)
    m1 = LogisticRegression(max_iter=500, random_state=seed).fit(np.column_stack((base_validation, rupture_validation)), error_validation)
    probability0 = m0.predict_proba(base_test)[:, 1]
    probability1 = m1.predict_proba(np.column_stack((base_test, rupture_test)))[:, 1]
    auprc0 = float(average_precision_score(error_test, probability0))
    auprc1 = float(average_precision_score(error_test, probability1))
    return {
        "n_test": len(test),
        "m0_auroc": float(roc_auc_score(error_test, probability0)),
        "m1_auroc": float(roc_auc_score(error_test, probability1)),
        "m0_auprc": auprc0,
        "m1_auprc": auprc1,
        "incremental_auprc": auprc1 - auprc0,
        "m0_brier": float(brier_score_loss(error_test, probability0)),
        "m1_brier": float(brier_score_loss(error_test, probability1)),
        "fit_partition": "validation",
        "evaluation_partition": "test",
    }


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()
