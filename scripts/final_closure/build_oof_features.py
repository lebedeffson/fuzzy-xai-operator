#!/usr/bin/env python3
"""Build predictive-only OOF features without reading sealed-test artifacts."""

from __future__ import annotations

import json
import math
import subprocess

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import ROOT, STUDY, sha256, write


DATA_ROOT = ROOT / "data/confirmatory"
OUTPUT = STUDY / "oof_features"
SEED = 7419
PREDICTIVE_CHANNELS = (
    "calibrated_confidence",
    "prediction_margin",
    "normalized_entropy",
    "calibration_residual",
    "boundary_distance",
    "model_disagreement",
    "shift_score",
    "rare_group_indicator",
    "missingness_profile",
)
ROUTE_CHANNELS = (
    "provenance_completeness",
    "typed_route_fault",
    "explainer_disagreement",
    "seed_stability",
    "bootstrap_stability",
    "perturbation_stability",
    "representation_class",
    "reduction_loss",
    "rule_redundancy",
    "conflict_severity",
    "missing_evidence_channels",
    "reference_set_deviation",
    "canonical_hash_status",
)


def main() -> None:
    if not (STUDY / "final_leakage_audit.json").is_file():
        raise SystemExit("BLOCKED: final leakage audit must pass before OOF generation")
    if json.loads((STUDY / "final_leakage_audit.json").read_text())["status"] != "pass":
        raise SystemExit("BLOCKED: final leakage audit is not PASS")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports = []
    reports.append(_tabular_oof("bank_marketing", "y"))
    reports.append(_tabular_oof("default_credit_clients", "target", drop=("ID",)))
    reports.append(_text_oof("sms_spam"))
    reports.append(_image_oof("shoulder_implant_xray"))
    reports.append(_har_oof("uci_har_smartphones"))
    manifest = {
        "schema_version": "1.0",
        "source_commit": _git_head(),
        "feature_source": "out_of_fold_train_development_only",
        "sealed_test_loaded": False,
        "predictive_channels": list(PREDICTIVE_CHANNELS),
        "route_channels": list(ROUTE_CHANNELS),
        "P0_status": "partial_predictive_oof_missing_model_disagreement_and_shift",
        "P1_status": "pending_real_route_explanation_features",
        "lock_status": "blocked_route_features_pending",
        "datasets": reports,
    }
    write(STUDY / "confirmatory_feature_manifest.json", manifest)
    print(f"PASS: final_oof_features datasets={len(reports)} p0=partial p1=pending test_loaded=false")


def _tabular_oof(dataset_id: str, target: str, *, drop: tuple[str, ...] = ()) -> dict[str, object]:
    train = pd.read_csv(DATA_ROOT / dataset_id / "processed/train.csv")
    development = pd.read_csv(DATA_ROOT / dataset_id / "processed/development.csv")
    frame = pd.concat((train, development), ignore_index=True)
    ids = frame.pop("object_id_hash").astype(str).to_numpy()
    y = frame.pop(target).astype(str).to_numpy()
    frame = frame.drop(columns=list(drop), errors="ignore")
    categorical = frame.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = [column for column in frame.columns if column not in categorical]
    transformer = ColumnTransformer(
        (
            (
                "numeric",
                Pipeline((("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()))),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    (("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore")))
                ),
                categorical,
            ),
        )
    )
    estimator = Pipeline((("features", transformer), ("model", LogisticRegression(max_iter=400, random_state=SEED))))
    missingness = frame.isna().mean(axis=1).to_numpy(dtype=float)
    return _run_oof(dataset_id, frame, y, ids, estimator, missingness=missingness)


def _text_oof(dataset_id: str) -> dict[str, object]:
    train = pd.read_csv(DATA_ROOT / dataset_id / "processed/train.csv")
    development = pd.read_csv(DATA_ROOT / dataset_id / "processed/development.csv")
    frame = pd.concat((train, development), ignore_index=True)
    estimator = Pipeline(
        (
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30_000, sublinear_tf=True)),
            ("model", LogisticRegression(max_iter=400, random_state=SEED)),
        )
    )
    return _run_oof(
        dataset_id,
        frame["text"].fillna("").astype(str).to_numpy(),
        frame["target"].astype(str).to_numpy(),
        frame["object_id_hash"].astype(str).to_numpy(),
        estimator,
        missingness=np.zeros(len(frame)),
    )


def _image_oof(dataset_id: str) -> dict[str, object]:
    x, y, ids, _ = _load_npz_train_development(dataset_id)
    features = x[:, ::4, ::4].reshape(len(x), -1).astype(np.float32) / 255.0
    estimator = Pipeline((("scale", StandardScaler()), ("model", LogisticRegression(max_iter=500, random_state=SEED))))
    return _run_oof(dataset_id, features, y.astype(str), ids.astype(str), estimator, missingness=np.zeros(len(x)))


def _har_oof(dataset_id: str) -> dict[str, object]:
    x, y, ids, groups = _load_npz_train_development(dataset_id)
    if groups is None:
        raise SystemExit("FAIL: HAR OOF requires subject groups")
    features = _timeseries_features(x)
    estimator = Pipeline((("scale", StandardScaler()), ("model", LogisticRegression(max_iter=500, random_state=SEED))))
    return _run_oof(
        dataset_id,
        features,
        y.astype(str),
        ids.astype(str),
        estimator,
        missingness=np.zeros(len(x)),
        groups=groups,
    )


def _run_oof(
    dataset_id: str,
    x,
    y: np.ndarray,
    ids: np.ndarray,
    estimator,
    *,
    missingness: np.ndarray,
    groups: np.ndarray | None = None,
) -> dict[str, object]:
    classes = np.asarray(sorted(set(y.tolist())))
    probabilities = np.zeros((len(y), len(classes)), dtype=np.float64)
    calibration_residual = np.zeros(len(y), dtype=np.float64)
    rare = np.zeros(len(y), dtype=np.float64)
    splitter = (
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        if groups is not None
        else StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    )
    split_iterator = splitter.split(np.zeros(len(y)), y, groups) if groups is not None else splitter.split(np.zeros(len(y)), y)
    fold_rows = []
    for fold, (fit_index, held_index) in enumerate(split_iterator):
        calibrated = CalibratedClassifierCV(clone(estimator), method="sigmoid", cv=3)
        calibrated.fit(_slice(x, fit_index), y[fit_index])
        fold_probabilities = calibrated.predict_proba(_slice(x, held_index))
        class_positions = {str(value): index for index, value in enumerate(calibrated.classes_)}
        for global_position, class_name in enumerate(classes):
            probabilities[held_index, global_position] = fold_probabilities[:, class_positions[str(class_name)]]
        fit_probabilities = calibrated.predict_proba(_slice(x, fit_index))
        calibration_residual[held_index] = _fit_fold_calibration_residual(
            fit_probabilities,
            y[fit_index],
            calibrated.classes_,
            fold_probabilities,
        )
        fit_counts = pd.Series(y[fit_index]).value_counts(normalize=True)
        held_predictions = calibrated.classes_[np.argmax(fold_probabilities, axis=1)]
        rare[held_index] = np.asarray(
            [float(fit_counts.get(value, 0.0) < 0.10) for value in held_predictions]
        )
        fold_rows.append({"fold": fold, "fit": len(fit_index), "held_out": len(held_index)})
    order = np.argsort(probabilities, axis=1)
    predicted_positions = order[:, -1]
    predictions = classes[predicted_positions]
    confidence = probabilities[np.arange(len(y)), predicted_positions]
    margin = confidence - probabilities[np.arange(len(y)), order[:, -2]] if len(classes) > 1 else confidence
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1) / math.log(len(classes))
    output = OUTPUT / f"{dataset_id}.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for index, object_id in enumerate(ids):
            predictive = (
                float(confidence[index]),
                float(margin[index]),
                float(entropy[index]),
                float(calibration_residual[index]),
                float(margin[index]),
                None,
                None,
                float(rare[index]),
                float(np.clip(missingness[index], 0.0, 1.0)),
            )
            row = {
                "dataset_id": dataset_id,
                "object_id_hash": str(object_id),
                "split_id": "train-development-oof",
                "source_is_oof": True,
                "true_label": str(y[index]),
                "predicted_label": str(predictions[index]),
                "predictive": dict(zip(PREDICTIVE_CHANNELS, predictive, strict=True)),
                "route": dict.fromkeys(ROUTE_CHANNELS),
                "missing_channels": [
                    name
                    for name, value in zip((*PREDICTIVE_CHANNELS, *ROUTE_CHANNELS), (*predictive, *((None,) * len(ROUTE_CHANNELS))), strict=True)
                    if value is None
                ],
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "dataset_id": dataset_id,
        "objects": len(y),
        "classes": classes.tolist(),
        "accuracy": float(accuracy_score(y, predictions)),
        "log_loss": float(log_loss(y, probabilities, labels=classes)),
        "folds": fold_rows,
        "artifact_path": output.relative_to(ROOT).as_posix(),
        "artifact_sha256": sha256(output),
        "model_family": "calibrated_logistic_baseline",
        "feature_leakage_controls": {
            "calibration_residual": "fit_fold_reliability_map_only",
            "rare_group_indicator": "predicted_class_prevalence_in_fit_fold",
            "held_out_label_used_as_feature": False,
        },
        "missing_predictive_channels": ["model_disagreement", "shift_score"],
        "route_features_status": "pending_real_explanation_artifacts",
    }


def _load_npz_train_development(dataset_id: str):
    payloads = []
    for name in ("train", "development"):
        with np.load(DATA_ROOT / dataset_id / f"processed/{name}.npz") as payload:
            payloads.append({key: payload[key].copy() for key in payload.files})
    x = np.concatenate([payload["x"] for payload in payloads])
    y = np.concatenate([payload["y"] for payload in payloads])
    ids = np.concatenate([payload["object_id_hash"] for payload in payloads])
    groups = np.concatenate([payload["subject_id"] for payload in payloads]) if "subject_id" in payloads[0] else None
    return x, y, ids, groups


def _timeseries_features(x: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            x.mean(axis=1),
            x.std(axis=1),
            x.min(axis=1),
            x.max(axis=1),
            np.mean(np.square(x), axis=1),
        ),
        axis=1,
    )


def _slice(value, indices: np.ndarray):
    return value.iloc[indices] if hasattr(value, "iloc") else value[indices]


def _fit_fold_calibration_residual(
    fit_probabilities: np.ndarray,
    fit_labels: np.ndarray,
    estimator_classes: np.ndarray,
    held_probabilities: np.ndarray,
) -> np.ndarray:
    """Map held-out confidence to a calibration gap learned on the fit fold."""
    fit_positions = np.argmax(fit_probabilities, axis=1)
    fit_predictions = estimator_classes[fit_positions]
    fit_confidence = fit_probabilities[np.arange(len(fit_probabilities)), fit_positions]
    fit_correct = (fit_predictions == fit_labels).astype(float)
    boundaries = np.linspace(0.0, 1.0, 11)
    bin_ids = np.clip(np.digitize(fit_confidence, boundaries[1:-1]), 0, 9)
    global_gap = abs(float(fit_correct.mean()) - float(fit_confidence.mean()))
    gaps = np.full(10, global_gap, dtype=np.float64)
    for bin_id in range(10):
        selected = bin_ids == bin_id
        if selected.any():
            gaps[bin_id] = abs(float(fit_correct[selected].mean()) - float(fit_confidence[selected].mean()))
    held_confidence = np.max(held_probabilities, axis=1)
    held_bins = np.clip(np.digitize(held_confidence, boundaries[1:-1]), 0, 9)
    return gaps[held_bins]


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


if __name__ == "__main__":
    main()
