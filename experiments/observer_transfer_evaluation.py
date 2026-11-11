"""Developer-specified transfer audit on previously unused sklearn datasets.

The companion JSON protocol is committed separately *before first execution*.
This is NOT external preregistration or independent blind validation.
Tests use genuine native SHAP ExactExplainer and independent model probes.
The baseline checks identity+background+score channel+efficiency, but does
not compare attribution values with the exact Shapley component oracle.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import shap
import sklearn
from fuzzyxai.selective_observer import (
    EXACT_INTERVENTIONAL_PROTOCOL,
    LocalEvidenceStatus,
    LocalExplanationEvidence,
    collect_exact_interventional_shapley,
    verify_exact_interventional_shapley,
)
from sklearn.datasets import load_diabetes, load_iris, load_linnerud
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from experiments.observer_local_evidence_benchmark import _canonical_local_digest

PROTOCOL_PATH = Path(__file__).with_name("observer_transfer_protocol_v1.json")
TOLERANCE = 1e-6


def _baseline(
    explanation: LocalExplanationEvidence, reference, *,
    tolerance: float = TOLERANCE,
) -> LocalEvidenceStatus:
    """Strong identity/score-space/background/efficiency, no component oracle."""
    if explanation.scope != "local":
        return LocalEvidenceStatus.INCOMPATIBLE
    if not explanation.attribution_protocol:
        return LocalEvidenceStatus.INSUFFICIENT_EVIDENCE
    if explanation.attribution_protocol != EXACT_INTERVENTIONAL_PROTOCOL:
        return LocalEvidenceStatus.INSUFFICIENT_EVIDENCE
    for name in ("object_id", "model_version", "target_class", "feature_names",
                 "input_sha256", "background_sha256", "score_channel"):
        expected = getattr(reference, name)
        actual = getattr(explanation, name)
        if actual is None:
            return LocalEvidenceStatus.INSUFFICIENT_EVIDENCE
        if actual != expected:
            return LocalEvidenceStatus.INCOMPATIBLE
    if explanation.contributions is None:
        return LocalEvidenceStatus.INSUFFICIENT_EVIDENCE
    if len(explanation.contributions) != len(reference.attributions):
        return LocalEvidenceStatus.INCOMPATIBLE
    if any(not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in explanation.contributions):
        return LocalEvidenceStatus.INCOMPATIBLE
    actual_sum = math.fsum(explanation.contributions)
    expected_sum = reference.explained_score - reference.baseline_score
    return (
        LocalEvidenceStatus.CERTIFIED
        if abs(actual_sum - expected_sum) <= tolerance
        else LocalEvidenceStatus.INCOMPATIBLE
    )


def _estimator(name: str, seed: int):
    if name == "iris_hgb":
        return HistGradientBoostingClassifier(
            max_iter=32, max_leaf_nodes=7, random_state=seed
        )
    if name == "diabetes_etr":
        return ExtraTreesRegressor(
            n_estimators=32, max_depth=7, random_state=seed
        )
    if name == "linnerud_knn":
        return KNeighborsRegressor(n_neighbors=3)
    raise ValueError(f"Unknown transfer task {name}")


def _data(name: str, columns: tuple[int, ...]):
    if name == "iris_hgb":
        ds = load_iris()
        target = np.asarray(ds.target, dtype=int)
    elif name == "diabetes_etr":
        ds = load_diabetes()
        target = np.asarray(ds.target, dtype=float)
    elif name == "linnerud_knn":
        ds = load_linnerud()
        target = np.asarray(ds.target, dtype=float)[:, 0]
    else:
        raise ValueError(f"Unknown transfer dataset: {name}")
    features = np.asarray(ds.data, dtype=float)[:, list(columns)]
    names = tuple(str(ds.feature_names[idx]) for idx in columns)
    return features, target, names


def _native_and_probe(
    *,
    input_row: np.ndarray,
    background: np.ndarray,
    object_id: str,
    version: str,
    target_class: str,
    score_channel: str,
    feature_names: tuple[str, ...],
    score_fn,
):
    masker = shap.maskers.Independent(background, max_samples=len(background))
    native = np.asarray(
        shap.ExactExplainer(score_fn, masker)(input_row.reshape(1, -1)).values[0],
        dtype=float,
    )
    oracle = collect_exact_interventional_shapley(
        object_id=object_id,
        model_version=version,
        target_class=target_class,
        score_channel=score_channel,
        feature_names=feature_names,
        input_values=tuple(float(v) for v in input_row),
        background_rows=tuple(tuple(float(v) for v in row) for row in background),
        target_scores=lambda rows: tuple(
            float(v) for v in score_fn(np.asarray(rows, dtype=float))
        ),
    )
    assert oracle.attributions is not None
    delta = max(abs(float(a) - float(b)) for a, b in zip(
        native, oracle.attributions, strict=True
    ))
    return tuple(float(v) for v in native), oracle, delta


def run_transfer(*, per_task: int | None = None) -> dict[str, object]:
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    if protocol["schema_version"] != "observer-transfer-developmental-v1":
        raise ValueError("unsupported transfer protocol")
    if protocol["absolute_comparison_tolerance"] != TOLERANCE:
        raise ValueError("comparison tolerance differs from frozen protocol")
    seed = int(protocol["seed"])
    count = (
        int(protocol["n_test_objects_per_task"])
        if per_task is None else per_task
    )
    if count < 1 or count > int(protocol["n_test_objects_per_task"]):
        raise ValueError("per_task must be within frozen sample count")
    rows: list[dict[str, object]] = []
    models: dict[str, object] = {}
    largest_native_gap = 0.0
    max_background_shift = 0.0
    for task in protocol["tasks"]:
        name = task["name"]
        cols = tuple(int(i) for i in task["feature_indices"])
        features, target, names = _data(name, cols)
        stratify = target if name == "iris_hgb" else None
        train_ids, test_ids = train_test_split(
            np.arange(len(target)),
            test_size=float(protocol["test_fraction"]),
            random_state=seed,
            stratify=stratify,
        )
        estimator = _estimator(name, seed)
        estimator.fit(features[train_ids], target[train_ids])
        if len(test_ids) < count or len(train_ids) < 12:
            raise ValueError("insufficient train/test rows for frozen protocol")
        background1 = features[train_ids[:6]]
        background2 = features[train_ids[6:12]]
        model_hash = hashlib.sha256(
            pickle.dumps(estimator, protocol=4)
        ).hexdigest()
        version = f"transfer-{name}-{model_hash[:16]}"
        models[name] = {
            "model": type(estimator).__name__,
            "fitted_model_pickle_sha256": model_hash,
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "train_ids_sha256": hashlib.sha256(train_ids.tobytes()).hexdigest(),
            "test_ids_sha256": hashlib.sha256(test_ids.tobytes()).hexdigest(),
        }

        for position, test_id in enumerate(test_ids[:count]):
            item = features[test_id]
            donor = features[test_ids[(position + 1) % count]]
            if name == "iris_hgb":
                probabilities = np.asarray(
                    estimator.predict_proba(item.reshape(1, -1))[0],
                    dtype=float,
                )
                target_index = int(np.argmax(probabilities))
                target_class = str(estimator.classes_[target_index])

                def scoring(batch, fitted=estimator, chosen=target_index):
                    return fitted.predict_proba(
                        np.asarray(batch, dtype=float)
                    )[:, chosen]
            else:
                target_class = "scalar_regression_output"

                def scoring(batch, fitted=estimator):
                    return fitted.predict(np.asarray(batch, dtype=float))

            score_channel = task["score_channel"]
            object_id = f"{name}-{test_id}"
            phi1, oracle1, gap1 = _native_and_probe(
                input_row=item,
                background=background1,
                object_id=object_id,
                version=version,
                target_class=target_class,
                score_channel=score_channel,
                feature_names=names,
                score_fn=scoring,
            )
            phi2, oracle2, gap2 = _native_and_probe(
                input_row=item,
                background=background2,
                object_id=object_id,
                version=version,
                target_class=target_class,
                score_channel=score_channel,
                feature_names=names,
                score_fn=scoring,
            )
            largest_native_gap = max(largest_native_gap, gap1, gap2)
            max_background_shift = max(
                max_background_shift,
                max(abs(a-b) for a, b in zip(phi1, phi2, strict=True)),
            )
            donor_phi, _, _ = _native_and_probe(
                input_row=donor,
                background=background1,
                object_id=object_id + "-donor",
                version=version,
                target_class=target_class,
                score_channel=score_channel,
                feature_names=names,
                score_fn=scoring,
            )
            width = len(names)
            # Align donor's *total* to the explained total so even the
            # strong structural+efficiency baseline cannot catch it.
            donor_offset = (math.fsum(phi1) - math.fsum(donor_phi)) / width
            donor_same_sum = tuple(value + donor_offset for value in donor_phi)

            local = LocalExplanationEvidence(
                object_id=object_id,
                model_version=version,
                target_class=target_class,
                scope="local",
                feature_names=names,
                contributions=phi1,
                attribution_protocol=EXACT_INTERVENTIONAL_PROTOCOL,
                score_channel=score_channel,
                input_sha256=oracle1.input_sha256,
                background_sha256=oracle1.background_sha256,
            )
            alternative = replace(
                local, contributions=phi2,
                background_sha256=oracle2.background_sha256,
            )
            changed = {
                "clean_primary": (local, oracle1),
                "clean_alternate_background": (alternative, oracle2),
                "zero_sum_0_05": (replace(
                    local,
                    contributions=tuple(
                        phi1[i] + (0.05 if i == 0 else -0.05 if i == 1 else 0.0)
                        for i in range(width)
                    ),
                ), oracle1),
                "zero_sum_subthreshold_1e_8": (replace(
                    local,
                    contributions=tuple(
                        phi1[i] + (1e-8 if i == 0 else -1e-8 if i == 1 else 0.0)
                        for i in range(width)
                    ),
                ), oracle1),
                "donor_attribution": (
                    replace(local, contributions=donor_same_sum), oracle1
                ),
                "wrong_background_sha": (
                    replace(local, background_sha256="f" * 64), oracle1
                ),
                "wrong_score_channel": (
                    replace(local, score_channel="decision_function"
                            if score_channel == "predict_proba"
                            else "predict_proba"), oracle1
                ),
                "approximation_protocol": (
                    replace(local, attribution_protocol="approximate_permutation_shap"), oracle1
                ),
                "missing_background_sha": (
                    replace(local, background_sha256=None), oracle1
                ),
            }
            if set(changed) != {
                spec["id"] for spec in protocol["fault_families"]
            }:
                raise ValueError("fault families diverge from frozen protocol")
            for fault, (mutant, measured) in changed.items():
                mutant = replace(mutant, source_sha256=_canonical_local_digest(mutant))
                observed = verify_exact_interventional_shapley(mutant, measured)
                baseline = _baseline(mutant, measured)
                distance = max(
                    abs(float(a)-float(b)) for a, b in zip(
                        mutant.contributions, measured.attributions, strict=True
                    )
                )
                rows.append({
                    "dataset_model": name,
                    "object_id": object_id,
                    "fault": fault,
                    "outcome": observed.status.value,
                    "baseline": baseline.value,
                    "distance": distance,
                })

    families = {}
    for family in sorted({row["fault"] for row in rows}):
        group = [r for r in rows if r["fault"] == family]
        families[family] = {
            "n": len(group),
            "certified": sum(r["outcome"] == "certified" for r in group),
            "incompatible": sum(r["outcome"] == "incompatible" for r in group),
            "insufficient_evidence": sum(r["outcome"] == "insufficient_evidence" for r in group),
            "baseline_certified": sum(r["baseline"] == "certified" for r in group),
            "max_component_error": max(r["distance"] for r in group),
        }
    return {
        "phase": "developer_specified_out_of_family_transfer",
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "n_distinct_test_objects": len(rows) // len(protocol["fault_families"]),
        "n_dependent_variants": len(rows),
        "by_family": families,
        "largest_clean_native_vs_oracle_error": largest_native_gap,
        "largest_valid_background_change_in_attribution": max_background_shift,
        "model_evidence": models,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "shap": shap.__version__,
        },
        "limitations": [
            "Not a blind independent evaluation; designer specified both faults and observer",
            "3 new sklearn task-model pairs, 5 held-out objects per task; variants are correlated",
            "Exact interventional game; genuine alternative backgrounds are valid controls",
            "The below-tolerance perturbation is intentionally undetectable",
            "No measured population false-block rate or unknown-fault reliability",
            "Hashes remain declarations; model callback authenticity is not cryptographic",
        ],
        "claim_allowed": False,
    }


if __name__ == "__main__":
    print(json.dumps(run_transfer(), ensure_ascii=False, indent=2, sort_keys=True))
