"""Held-out WDBC study of distinct IT2 FOU and split-conformal safeguards.

Five train/calibration/test splits, paired Type-1 and IT2 comparisons.
The results measure *empirical* set coverage and selective risk; split
conformal theory applies only if exchangeability and data-separation
assumptions hold. The observed fractions are not clinical guarantees.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from fuzzyxai.interval_type2 import content_sha256
from fuzzyxai.it2_conformal import IT2SplitConformal
from .doctoral_interval_type2 import NativeIntervalType2Logistic


def _id(index: int) -> str:
    return f"wdbc-sklearn-row-{int(index)}"


def _metrics(
    true_labels: np.ndarray, predicted_probability: np.ndarray,
    sets: list[dict[str, Any]], *, extra_accept: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    n = len(true_labels)
    observed = np.asarray(
        [int(true_labels[i]) in row["label_set"] for i, row in enumerate(sets)], dtype=bool)
    sizes = np.asarray([row["set_size"] for row in sets], dtype=int)
    preds = (predicted_probability >= 0.5).astype(int)
    singletons = np.asarray(
        [len(row["label_set"]) == 1 and row["label_set"][0] == int(preds[i])
         for i, row in enumerate(sets)], dtype=bool)
    auto = singletons if extra_accept is None else (singletons & extra_accept)
    accepted = int(auto.sum())
    errors = int(np.sum((preds != true_labels) & auto))
    malignant_false_negatives = int(np.sum((true_labels == 1) & (preds == 0) & auto))
    return {
        "n_test": n,
        "empirical_marginal_coverage": float(observed.mean()),
        "singleton_set_rate": float(np.mean(sizes == 1)),
        "empty_set_rate": float(np.mean(sizes == 0)),
        "both_labels_rate": float(np.mean(sizes == 2)),
        "mean_set_size": float(sizes.mean()),
        "automatic_accept_count": accepted,
        "automatic_accept_fraction": accepted / n,
        "selective_error": errors / accepted if accepted else None,
        "automatically_accepted_false_negatives": malignant_false_negatives,
    }


def run_one(
    seed: int, out_dir: Path, *, n_features: int = 5,
    n_bootstrap: int = 16, alpha: float = 0.1,
) -> dict[str, Any]:
    data = load_breast_cancer()
    x = np.asarray(data.data, dtype=float)
    y = (np.asarray(data.target) == 0).astype(int)  # positive = malignant
    identities = np.arange(len(y))
    x_tr, x_other, y_tr, y_other, id_tr, id_other = train_test_split(
        x, y, identities, test_size=0.4, stratify=y, random_state=seed)
    x_val, x_te, y_val, y_te, id_val, id_te = train_test_split(
        x_other, y_other, id_other, test_size=0.5,
        stratify=y_other, random_state=seed + 1)
    # Policy thresholds are tuned on a fourth disjoint partition, not
    # calibration or test, so equal coverage is not manufactured by test labels.
    x_cal, x_policy, y_cal, y_policy, id_cal, id_policy = train_test_split(
        x_val, y_val, id_val, test_size=0.30,
        stratify=y_val, random_state=seed + 2)
    groups = [set(map(int, g)) for g in (id_tr, id_cal, id_policy, id_te)]
    if not all(groups[i].isdisjoint(groups[j])
               for i in range(len(groups)) for j in range(i+1, len(groups))) or (
               sum(map(len, groups)) != len(y)):
        raise AssertionError("train / calibration / policy / test groups overlap")
    train_ids = [_id(v) for v in id_tr]
    calibration_ids = [_id(v) for v in id_cal]
    policy_ids = [_id(v) for v in id_policy]
    test_ids = [_id(v) for v in id_te]
    native = NativeIntervalType2Logistic(
        n_features=n_features, n_bootstrap=n_bootstrap, seed=seed)
    native.fit(x_tr, y_tr, list(data.feature_names))
    if native.it2 is None:
        raise AssertionError("type-2 model did not fit")
    p_val_it2 = native.predict_proba(x_cal)[:, 1]
    p_policy_it2 = native.predict_proba(x_policy)[:, 1]
    p_test_it2 = native.predict_proba(x_te)[:, 1]
    p_val_t1 = native.type1.predict_proba(x_cal)[:, 1]
    p_policy_t1 = native.type1.predict_proba(x_policy)[:, 1]
    p_test_t1 = native.type1.predict_proba(x_te)[:, 1]
    it2_calibration = IT2SplitConformal.fit_it2(
        native.it2, x_cal.tolist(), y_cal.tolist(),
        training_ids=train_ids, calibration_ids=calibration_ids,
        calibration_dataset_version=f"sklearn:wdbc:seed={seed}:calibration",
        alpha=alpha)
    # The Type-1 comparator uses the identical train/calibration/test split
    # and the same nonconformity score, with its own held-out quantile.
    t1_calibration = IT2SplitConformal.fit_probabilities(
        p_val_t1.tolist(), y_cal.tolist(),
        model_sha256=content_sha256(native.type1.fitted_json()),
        training_ids=train_ids, calibration_ids=calibration_ids,
        calibration_dataset_version=f"sklearn:wdbc:seed={seed}:calibration",
        alpha=alpha,
    )
    if not np.allclose(p_val_it2, [
        native.it2.infer(row)["probability_midpoint"] for row in x_cal
    ], rtol=0, atol=1e-10):
        raise AssertionError("published IT2 calibration probabilities differ from inference")
    sets_it2 = [it2_calibration.predict_set(float(p), object_id=oid)
                for p, oid in zip(p_test_it2, test_ids, strict=True)]
    sets_t1 = [t1_calibration.predict_set(float(p), object_id=oid)
               for p, oid in zip(p_test_t1, test_ids, strict=True)]
    fuzzy_actions = [native.decision(row) for row in x_te]
    fuzzy_accept = np.asarray([bool(v["accepted"]) for v in fuzzy_actions], dtype=bool)

    # Coverage matching is decided from a separate policy subset with NO
    # access to test labels, while both conformal quantiles use calibration ONLY.
    policy_sets_it2 = [
        it2_calibration.predict_set(float(p), object_id=oid)
        for p, oid in zip(p_policy_it2, policy_ids, strict=True)
    ]
    policy_sets_t1 = [
        t1_calibration.predict_set(float(p), object_id=oid)
        for p, oid in zip(p_policy_t1, policy_ids, strict=True)
    ]
    policy_fou_accept = np.asarray(
        [bool(native.decision(row)["accepted"]) for row in x_policy], dtype=bool)
    policy_t2_singleton = np.asarray([
        result["set_size"] == 1 and result["label_set"][0] == int(p >= 0.5)
        for result, p in zip(policy_sets_it2, p_policy_it2, strict=True)
    ], dtype=bool)
    target_coverage = float(np.mean(policy_fou_accept & policy_t2_singleton))
    policy_t1_singleton = np.asarray([
        result["set_size"] == 1 and result["label_set"][0] == int(p >= 0.5)
        for result, p in zip(policy_sets_t1, p_policy_t1, strict=True)
    ], dtype=bool)
    policy_margins = np.abs(p_policy_t1 - 0.5)
    threshold_candidates = [-1.0] + sorted(set(float(x) for x in policy_margins))
    matched_threshold = min(
        threshold_candidates,
        key=lambda t: (abs(float(np.mean(policy_t1_singleton & (policy_margins > t)))
                           - target_coverage), -t),
    )
    test_margins = np.abs(p_test_t1 - 0.5)
    type1_matched_abstention = test_margins > matched_threshold
    rows: list[dict[str, Any]] = []
    for i, oid in enumerate(test_ids):
        rows.append({
            "row_id": oid,
            "true_label_malignant": int(y_te[i]),
            "p_type1": float(p_test_t1[i]),
            "p_it2_point": float(p_test_it2[i]),
            "it2_fou_action": fuzzy_actions[i]["action"],
            "t1_conformal_label_set": "|".join(map(str, sets_t1[i]["label_set"])),
            "it2_conformal_label_set": "|".join(map(str, sets_it2[i]["label_set"])),
            "t1_true_in_set": int(int(y_te[i]) in sets_t1[i]["label_set"]),
            "it2_true_in_set": int(int(y_te[i]) in sets_it2[i]["label_set"]),
            "it2_combined_accept": int(fuzzy_accept[i] and sets_it2[i]["set_size"] == 1
                                       and sets_it2[i]["label_set"][0] == int(p_test_it2[i] >= 0.5)),
            "type1_coverage_matched_accept": int(
                type1_matched_abstention[i]
                and sets_t1[i]["set_size"] == 1
                and sets_t1[i]["label_set"][0] == int(p_test_t1[i] >= 0.5)),
        })
    report = {
        "schema_version": "fuzzyxai-wdbc-it2-vs-type1-split-conformal-v1",
        "status": "measured_on_held_out_rows",
        "seed": seed,
        "alpha": alpha,
        "calibration_n": int(len(y_cal)),
        "policy_selection_n": int(len(y_policy)),
        "policy_target_accept_fraction": target_coverage,
        "type1_policy_matched_margin_threshold": float(matched_threshold),
        "training_n": int(len(y_tr)),
        "test_n": int(len(y_te)),
        "model_sha256_it2": content_sha256(native.it2.to_dict()),
        "model_sha256_type1": content_sha256(native.type1.fitted_json()),
        "split_sha256": {
            "training": content_sha256(sorted(train_ids)),
            "calibration": content_sha256(sorted(calibration_ids)),
            "policy": content_sha256(sorted(policy_ids)),
            "test": content_sha256(sorted(test_ids)),
        },
        "it2_conformal": _metrics(y_te, p_test_it2, sets_it2, extra_accept=fuzzy_accept),
        "type1_conformal": _metrics(y_te, p_test_t1, sets_t1),
        "type1_conformal_coverage_matched": _metrics(
            y_te, p_test_t1, sets_t1, extra_accept=type1_matched_abstention),
        "it2_conformal_only": _metrics(y_te, p_test_it2, sets_it2),
        "it2_fou_only": {
            "accepted": int(fuzzy_accept.sum()), "coverage": float(fuzzy_accept.mean())
        },
        "calibration_receipts": {
            "it2": it2_calibration.receipt_sha256,
            "type1": t1_calibration.receipt_sha256,
        },
        "limitations": [
            "Five split seeds reuse the same source dataset; reported averages are descriptive, not independent replications",
            "Marginal conformal coverage requires model fixed before calibration and exchangeable test/calibration scores",
            "Marginal guarantee does NOT imply patient-level, conditional, under-shift, or clinical safety",
            "Prediction sets are calibrated for Type-1/IT2 point probabilities, NOT for membership-FOU widths",
            "IT2 coefficients learned from nominal Type-1 rules; membership uncertainty estimated from training-only bootstrap",
            "Type-1 abstention threshold selected on a separate policy subset; realized test acceptance need not exactly match IT2",
            "Calibration and policy selection are stratified by labels; exact exchangeability assumptions are not established by this design",
            "Automatic acceptance coverage and empirical error are descriptive performance metrics, not a safety proof",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "calibration_it2.json").write_text(json.dumps(it2_calibration.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "calibration_type1.json").write_text(json.dumps(t1_calibration.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "heldout_conformal.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def run(root: Path, *, seeds: tuple[int, ...] = (17, 42, 73, 123, 2026),
        n_bootstrap: int = 16, alpha: float = 0.1) -> dict[str, Any]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("unique nonempty random seed list required")
    reports = [
        run_one(s, root / f"seed_{s}", n_bootstrap=n_bootstrap, alpha=alpha)
        for s in seeds
    ]
    aggregate: dict[str, dict[str, float | None]] = {}
    for name in ("it2_conformal", "type1_conformal", "type1_conformal_coverage_matched", "it2_conformal_only"):
        aggregate[name] = {}
        for key in ("empirical_marginal_coverage", "automatic_accept_fraction"):
            values = [float(row[name][key]) for row in reports]
            aggregate[name][key + "_mean"] = mean(values)
            aggregate[name][key + "_std_descriptive"] = (
                stdev(values) if len(values) > 1 else None)
    summary = {
        "schema_version": "fuzzyxai-it2-conformal-study-v1",
        "alpha": alpha,
        "seeds": list(seeds),
        "n_bootstrap": n_bootstrap,
        "aggregates_descriptive": aggregate,
        "per_split": reports,
        "scope": "exploratory_measured_empirical_marginal_coverage_under_holdout",
        "independence_warning": "Repeated splits of one dataset are not independent clinical validations",
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest_sha256.json"}
    (root / "manifest_sha256.json").write_text(
        json.dumps({"schema": "sha256-manifest-v1", "files": manifest}, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/chapter5/it2_conformal"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73, 123, 2026])
    parser.add_argument("--bootstrap", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    print(json.dumps(run(args.out, seeds=tuple(args.seeds),
                         n_bootstrap=args.bootstrap, alpha=args.alpha),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
