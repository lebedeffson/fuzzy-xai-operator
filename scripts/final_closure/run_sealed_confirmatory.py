#!/usr/bin/env python3
"""Run the one-shot sealed H3/H7 confirmatory contour after protocol lock."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from common import ROOT, STUDY, load, sha256, write
from confirmatory_experiments import run_label_free_experiments, score_h6b
from oof_pipeline import (
    DATASETS,
    DATA_ROOT,
    PREDICTIVE_CHANNELS,
    ROUTE_CHANNELS,
    SEED,
    LoadedData,
    NeuralArrayAdapter,
    TabularAdapter,
    TextAdapter,
    _calibration_gap,
    _model_hash,
    _occlusion_contributions,
    _profile_scores,
    _route_channels,
    _sample,
)
from run_real_formative import (
    BUDGETS,
    COMPONENT_GROUPS,
    PRIMARY_BUDGET,
    _baseline_scores,
    _load_rows,
    _matrices,
    _operational_target,
)


OUTPUT = STUDY / "confirmatory"
LOCK = STUDY / "confirmatory_protocol_lock.json"
OPENING = STUDY / "confirmatory_opening_record.json"
COMPLETION = STUDY / "confirmatory_completion_marker.json"
INVALID = STUDY / "confirmatory_invalid_marker.json"
KEY_PATH = Path.home() / ".config/fuzzyxai/confirmatory_vault_aes256.pass"


def main() -> None:
    _require_locked_clean_state()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prescore_manifest = _build_prescore()
    write(OUTPUT / "prescore_manifest.json", prescore_manifest)
    write(
        OPENING,
        {
            "status": "labels_opening_for_scoring",
            "protocol_lock_sha256": sha256(LOCK),
            "prescore_manifest_sha256": sha256(OUTPUT / "prescore_manifest.json"),
            "post_open_tuning_forbidden": True,
        },
    )
    try:
        labels = _open_all_vaults()
        result = _score_preserved_actions(labels, prescore_manifest)
        write(OUTPUT / "h3_h7_summary.json", result)
        write(
            COMPLETION,
            {
                "status": "completed_once",
                "protocol_lock_sha256": sha256(LOCK),
                "opening_record_sha256": sha256(OPENING),
                "result_sha256": sha256(OUTPUT / "h3_h7_summary.json"),
                "post_open_tuning": False,
            },
        )
    except BaseException as error:
        write(
            INVALID,
            {
                "status": "invalid_after_label_opening",
                "protocol_lock_sha256": sha256(LOCK),
                "error_type": type(error).__name__,
                "retry_as_original_confirmatory_forbidden": True,
            },
        )
        raise
    print(f"PASS: final_sealed_confirmatory objects={result['objects']} opened_once=true post_open_tuning=false")


def _require_locked_clean_state() -> None:
    if not LOCK.is_file() or load(LOCK).get("status") != "locked":
        raise SystemExit("BLOCKED: final confirmatory protocol is not locked")
    if COMPLETION.is_file():
        raise SystemExit("BLOCKED: confirmatory run already completed")
    if OPENING.is_file() or INVALID.is_file():
        raise SystemExit("BLOCKED: labels were already opened; a second original confirmatory run is forbidden")
    lock = load(LOCK)
    if lock.get("source_commit") != _git_head():
        raise SystemExit("BLOCKED: repository HEAD differs from the locked source commit")
    if not KEY_PATH.is_file() or os.stat(KEY_PATH).st_mode & 0o077:
        raise SystemExit("BLOCKED: private label-vault key is absent or has unsafe permissions")


def _build_prescore() -> dict[str, object]:
    feature_paths, evidence_paths, model_paths = [], [], []
    for dataset_id in DATASETS:
        features, evidence, model = _build_dataset_prescore(dataset_id)
        feature_paths.append(_artifact(features))
        evidence_paths.append(_artifact(evidence))
        model_paths.append(_artifact(model))
    feature_frame = _load_test_feature_rows()
    oof = _load_rows()
    oof_target, _ = _operational_target(oof)
    oof_p0, oof_p1, names = _matrices(oof)
    test_p0, test_p1, test_names = _matrices(feature_frame)
    if names != test_names:
        raise RuntimeError("P0/P1 feature schema changed between OOF and sealed test")
    p0_model = _fit_risk_model(oof_p0, oof_target)
    p1_model = _fit_risk_model(oof_p1, oof_target)
    p0_score = p0_model.predict_proba(test_p0)[:, 1]
    p1_score = p1_model.predict_proba(test_p1)[:, 1]
    scores = _baseline_scores(feature_frame, p0_score, p1_score)
    scores["route_risk_only"] = np.maximum.reduce(
        (
            feature_frame["typed_route_fault"].fillna(0).to_numpy(float),
            1.0 - feature_frame["provenance_completeness"].fillna(0).to_numpy(float),
            feature_frame["explainer_disagreement"].fillna(0).to_numpy(float),
        )
    )
    for component, component_names in COMPONENT_GROUPS.items():
        removed = set(component_names) | {f"{name}__missing" for name in component_names}
        keep = [index for index, name in enumerate(names) if name not in removed]
        model = _fit_risk_model(oof_p1[:, keep], oof_target)
        scores[f"P1_minus_{component}"] = model.predict_proba(test_p1[:, keep])[:, 1]
    policy_path = OUTPUT / "prescore_policy_actions.parquet"
    _write_prescore_actions(feature_frame, scores, policy_path)
    model_path = OUTPUT / "controller_models.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({"P0": p0_model, "P1": p1_model, "feature_names": names}, handle, protocol=5)
    auxiliary = run_label_free_experiments()
    return {
        "schema_version": "1.0",
        "phase": "sealed_test_prescore_without_labels",
        "protocol_lock_sha256": sha256(LOCK),
        "sealed_test_labels_loaded": False,
        "objects": len(feature_frame),
        "feature_artifacts": feature_paths,
        "canonical_evidence_artifacts": evidence_paths,
        "predictive_model_artifacts": model_paths,
        "controller_models": _artifact(model_path),
        "policy_actions": _artifact(policy_path),
        "label_free_experiments": auxiliary,
        "primary_review_budget": PRIMARY_BUDGET,
    }


def _build_dataset_prescore(dataset_id: str) -> tuple[Path, Path, Path]:
    training, test = _load_training_test(dataset_id)
    adapter, classes = training.adapter, np.asarray(sorted(set(training.y.tolist())))
    fit_index = np.arange(len(training.y))
    reference = adapter.reference(fit_index)
    roles = ("primary", "alternate", "seed", "bootstrap")
    models = {
        role: adapter.fit_model(fit_index, role=role, seed=SEED + index * 17)
        for index, role in enumerate(roles)
    }
    test_values = test.adapter.x
    probabilities = {role: adapter.predict_proba(model, test_values, classes) for role, model in models.items()}
    positions = np.argmax(probabilities["primary"], axis=1)
    predictions = classes[positions]
    contributions = {
        role: _occlusion_contributions(adapter, model, test_values, predictions, classes, reference)
        for role, model in models.items()
    }
    perturbed = adapter.perturb(test_values, reference, seed=SEED)
    contributions["perturbation"] = _occlusion_contributions(
        adapter, models["primary"], perturbed, predictions, classes, reference
    )
    primary = probabilities["primary"]
    confidence = primary[np.arange(len(primary)), positions]
    order = np.argsort(primary, axis=1)
    margin = confidence - primary[np.arange(len(primary)), order[:, -2]] if len(classes) > 1 else confidence
    entropy = -np.sum(primary * np.log(np.clip(primary, 1e-12, 1.0)), axis=1) / max(math.log(len(classes)), 1.0)
    calibrated, calibration_gap = _calibrate_from_oof(dataset_id, margin)
    disagreement = 0.5 * np.sum(np.abs(primary - probabilities["alternate"]), axis=1)
    fit_prediction = classes[np.argmax(adapter.predict_proba(models["primary"], training.adapter.x, classes), axis=1)]
    frequencies = pd.Series(fit_prediction).value_counts(normalize=True)
    rare = np.asarray([float(frequencies.get(value, 0.0) < 0.10) for value in predictions])
    fit_profile, _ = adapter.profile(training.adapter.x)
    test_profile, missingness = adapter.profile(test_values)
    shift, quality = _profile_scores(fit_profile, test_profile, missingness)
    route = _route_channels(contributions, entropy, shift)
    feature_path = OUTPUT / f"features/{dataset_id}.jsonl"
    evidence_path = OUTPUT / f"canonical/{dataset_id}.jsonl"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with feature_path.open("w", encoding="utf-8") as feature_handle, evidence_path.open("w", encoding="utf-8") as evidence_handle:
        for index, object_id in enumerate(test.object_ids):
            source = {
                "schema_version": "1.0",
                "dataset_id": dataset_id,
                "object_id_hash": str(object_id),
                "fold": "sealed-test-final-model",
                "model_id": adapter.primary_model_id,
                "explainer_id": adapter.explainer_id,
                "prediction": str(predictions[index]),
                "components": [
                    adapter.component_detail(component, _sample(test_values[index : index + 1]))
                    | {"contribution": float(contributions["primary"][index, component])}
                    for component in range(len(adapter.components))
                ],
            }
            canonical = json.dumps(source, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(canonical.encode()).hexdigest()
            evidence_handle.write(canonical + "\n")
            predictive_values = (
                calibrated[index], margin[index], entropy[index], calibration_gap[index], margin[index],
                disagreement[index], shift[index], rare[index], missingness[index], quality[index],
            )
            row = {
                "dataset_id": dataset_id,
                "modality": training.modality,
                "object_id_hash": str(object_id),
                "predicted_label": str(predictions[index]),
                "model_id": adapter.primary_model_id,
                "explainer_id": adapter.explainer_id,
                "canonical_evidence_sha256": digest,
                "predictive": dict(zip(PREDICTIVE_CHANNELS, map(_unit, predictive_values), strict=True)),
                "route": {
                    name: None if np.isnan(route[name][index]) else _unit(route[name][index]) for name in ROUTE_CHANNELS
                },
            }
            feature_handle.write(json.dumps(row, sort_keys=True) + "\n")
    model_path = OUTPUT / f"models/{dataset_id}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(models, handle, protocol=5)
    write(
        OUTPUT / f"models/{dataset_id}.manifest.json",
        {
            "dataset_id": dataset_id,
            "training_scope": "train_plus_development_before_test_label_opening",
            "model_hashes": {role: _model_hash(model) for role, model in models.items()},
            "artifact_sha256": sha256(model_path),
        },
    )
    print(f"PASS: sealed_prescore dataset={dataset_id} objects={len(test.object_ids)} labels_loaded=false", flush=True)
    return feature_path, evidence_path, model_path


def _load_training_test(dataset_id: str) -> tuple[LoadedData, LoadedData]:
    from oof_pipeline import _load

    training = _load(dataset_id)
    if dataset_id in {"bank_marketing", "default_credit_clients", "sms_spam"}:
        frame = pd.read_csv(DATA_ROOT / dataset_id / "processed/sealed_test.csv")
        ids = frame.pop("object_id_hash").astype(str).to_numpy()
        if dataset_id == "default_credit_clients":
            frame = frame.drop(columns=["ID"], errors="ignore")
        if dataset_id == "sms_spam":
            x: Any = frame["text"].fillna("").astype(str).to_numpy(dtype=object)
            adapter = TextAdapter(x, np.asarray([], dtype=str))
            modality = "text"
        else:
            x = frame
            adapter = TabularAdapter(x, np.asarray([], dtype=str))
            modality = "tabular"
    else:
        with np.load(DATA_ROOT / dataset_id / "processed/sealed_test.npz") as payload:
            x = payload["x"].copy()
            ids = payload["object_id_hash"].astype(str)
            groups = payload["subject_id"].copy() if "subject_id" in payload.files else None
        modality = "image" if dataset_id == "shoulder_implant_xray" else "timeseries"
        adapter = NeuralArrayAdapter(x, np.asarray([], dtype=str), modality=modality)
        return training, LoadedData(dataset_id, modality, x, np.asarray([], dtype=str), ids, groups, np.asarray([]), adapter)
    return training, LoadedData(dataset_id, modality, x, np.asarray([], dtype=str), ids, None, np.asarray([]), adapter)


def _calibrate_from_oof(dataset_id: str, test_margin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    path = STUDY / f"oof_features/{dataset_id}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["partition"] == "development":
            rows.append(row)
    margin = np.asarray([row["predictive"]["prediction_margin"] for row in rows], dtype=float)
    correct = np.asarray([row["predicted_label"] == row["true_label"] for row in rows], dtype=int)
    method = load(STUDY / f"dataset_manifests/{dataset_id}/model_manifest.json")["calibration"]["selected_method"]
    calibrator = _fit_calibrator(method, margin, correct)
    return calibrator(test_margin), _calibration_gap(margin, correct, test_margin)


def _fit_calibrator(method: str, values: np.ndarray, correct: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    values = np.clip(values, 1e-6, 1.0 - 1e-6)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip").fit(values, correct)
        return lambda x: np.clip(model.predict(np.clip(x, 1e-6, 1.0 - 1e-6)), 0.0, 1.0)
    if method == "temperature":
        logits = np.log(values / (1.0 - values))

        def objective(log_temperature: float) -> float:
            probability = 1.0 / (1.0 + np.exp(-logits / math.exp(log_temperature)))
            return float(-np.mean(correct * np.log(probability + 1e-12) + (1 - correct) * np.log(1 - probability + 1e-12)))

        temperature = math.exp(float(minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded").x))
        return lambda x: 1.0 / (1.0 + np.exp(-np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6))) / temperature))
    if method == "conformal":
        boundaries = np.quantile(values, np.linspace(0.0, 1.0, 11))
        bins = np.clip(np.digitize(values, boundaries[1:-1]), 0, 9)
        global_rate = float(np.mean(correct))
        rates = np.asarray([float(np.mean(correct[bins == i])) if np.any(bins == i) else global_rate for i in range(10)])
        return lambda x: rates[np.clip(np.digitize(x, boundaries[1:-1]), 0, 9)]
    model = LogisticRegression(max_iter=500).fit(values[:, None], correct)
    return lambda x: model.predict_proba(np.asarray(x)[:, None])[:, 1]


def _fit_risk_model(matrix: np.ndarray, target: np.ndarray) -> LogisticRegression:
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED).fit(matrix, target)


def _load_test_feature_rows() -> pd.DataFrame:
    rows = []
    for dataset_id in DATASETS:
        for line in (OUTPUT / f"features/{dataset_id}.jsonl").read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            row = {key: payload[key] for key in ("dataset_id", "modality", "object_id_hash", "predicted_label")}
            row.update(payload["predictive"])
            row.update(payload["route"])
            rows.append(row)
    return pd.DataFrame(rows)


def _write_prescore_actions(frame: pd.DataFrame, scores: dict[str, np.ndarray], path: Path) -> None:
    rows = []
    for budget in BUDGETS:
        review_count = int(round(budget * len(frame)))
        for policy, values in scores.items():
            if policy == "always_review":
                review = np.ones(len(frame), dtype=bool)
            elif policy == "always_accept":
                review = np.zeros(len(frame), dtype=bool)
            else:
                review = np.zeros(len(frame), dtype=bool)
                if review_count:
                    review[np.argsort(values, kind="stable")[-review_count:]] = True
            for index in range(len(frame)):
                rows.append(
                    {
                        "dataset_id": frame.iloc[index]["dataset_id"],
                        "object_id_hash": frame.iloc[index]["object_id_hash"],
                        "policy": policy,
                        "review_budget": budget,
                        "risk_score": float(values[index]),
                        "action": "review" if review[index] else "accept",
                    }
                )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _open_all_vaults() -> dict[str, str]:
    output: dict[str, str] = {}
    for dataset_id in DATASETS:
        vault = DATA_ROOT / dataset_id / "manifests/confirmatory_label_vault.enc"
        with tempfile.NamedTemporaryFile() as clear:
            subprocess.run(
                [
                    "openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-iter", "200000",
                    "-in", str(vault), "-out", clear.name, "-pass", f"file:{KEY_PATH}",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            clear.seek(0)
            labels = _parse_vault_payload(json.load(clear), dataset_id)
        expected = {
            json.loads(line)["object_id_hash"]
            for line in (OUTPUT / f"features/{dataset_id}.jsonl").read_text(encoding="utf-8").splitlines()
        }
        if set(labels) != expected:
            raise RuntimeError(
                f"sealed identity mismatch for {dataset_id}: features={len(expected)} labels={len(labels)}"
            )
        overlap = set(output) & set(labels)
        if overlap:
            raise RuntimeError(f"duplicate sealed identity across label vaults: {len(overlap)}")
        output.update({str(key): str(value) for key, value in labels.items()})
    return output


def _parse_vault_payload(payload: object, dataset_id: str) -> dict[str, str]:
    if not isinstance(payload, dict) or set(payload) != {"labels"} or not isinstance(payload["labels"], dict):
        raise RuntimeError(f"invalid label-vault envelope for {dataset_id}")
    return {str(key): str(value) for key, value in payload["labels"].items()}


def _score_preserved_actions(labels: dict[str, str], manifest: dict[str, object]) -> dict[str, object]:
    actions = pd.read_parquet(ROOT / manifest["policy_actions"]["path"])
    identities = set(actions["object_id_hash"].astype(str))
    if identities != set(labels):
        raise RuntimeError(f"sealed label identity mismatch: actions={len(identities)} labels={len(labels)}")
    feature_frame = _load_test_feature_rows()
    feature_frame["true_label"] = feature_frame["object_id_hash"].map(labels)
    feature_frame["prediction_failure"] = feature_frame["predicted_label"] != feature_frame["true_label"]
    target, reasons = _operational_target(feature_frame)
    target_by_id = dict(zip(feature_frame["object_id_hash"], target, strict=True))
    actions["invalid"] = actions["object_id_hash"].map(target_by_id).astype(bool)
    actions["invalid_accept"] = actions["invalid"] & (actions["action"] == "accept")
    aggregate = (
        actions.groupby(["policy", "review_budget"], as_index=False)
        .agg(
            invalid_automatic_actions=("invalid_accept", "sum"),
            review_rate=("action", lambda values: float(np.mean(values == "review"))),
            n=("object_id_hash", "size"),
        )
    )
    aggregate["automatic_coverage"] = 1.0 - aggregate["review_rate"]
    raw_path = OUTPUT / "scored_policy_results.parquet"
    actions.to_parquet(raw_path, index=False)
    table_path = OUTPUT / "policy_summary.parquet"
    aggregate.to_parquet(table_path, index=False)
    primary = aggregate[np.isclose(aggregate["review_budget"], PRIMARY_BUDGET)]
    p1 = primary[primary["policy"] == "full_fuzzyxai_P1"].iloc[0]
    protocol = load(STUDY / "protocol.json")
    baseline_name = protocol["primary_comparator_policy"]
    baseline = primary[primary["policy"] == baseline_name].iloc[0]
    reduction = (baseline["invalid_automatic_actions"] - p1["invalid_automatic_actions"]) / max(1, baseline["invalid_automatic_actions"])
    h7_count, h7_preserved = _verify_test_canonical()
    h6b = score_h6b(labels)
    return {
        "schema_version": "1.0",
        "phase": "sealed_confirmatory_scored_once",
        "protocol_lock_sha256": sha256(LOCK),
        "objects": len(feature_frame),
        "dataset_ids": list(DATASETS),
        "H3_P1": {
            "primary_review_budget": PRIMARY_BUDGET,
            "baseline": baseline_name,
            "baseline_invalid_automatic_actions": int(baseline["invalid_automatic_actions"]),
            "fuzzyxai_invalid_automatic_actions": int(p1["invalid_automatic_actions"]),
            "relative_reduction": float(reduction),
            "threshold_met_before_inference": bool(reduction >= 0.15),
        },
        "H7_A": {
            "artifacts": h7_count,
            "canonical_hash_preservation_rate": h7_preserved / max(1, h7_count),
        },
        "H6_B": h6b,
        "label_free_experiments": manifest["label_free_experiments"],
        "target_counts": {name: int(values.sum()) for name, values in reasons.items()} | {"any_invalid": int(target.sum())},
        "raw_results": _artifact(raw_path),
        "summary_table": _artifact(table_path),
        "post_open_tuning": False,
        "statistical_claim_status": "pending_final_statistics",
    }


def _verify_test_canonical() -> tuple[int, int]:
    total = preserved = 0
    for dataset_id in DATASETS:
        features = OUTPUT / f"features/{dataset_id}.jsonl"
        evidence = OUTPUT / f"canonical/{dataset_id}.jsonl"
        with features.open(encoding="utf-8") as feature_handle, evidence.open(encoding="utf-8") as evidence_handle:
            for feature_line, evidence_line in zip(feature_handle, evidence_handle, strict=True):
                feature, payload = json.loads(feature_line), json.loads(evidence_line)
                digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                total += 1
                preserved += digest == feature["canonical_evidence_sha256"]
    return total, preserved


def _artifact(path: Path) -> dict[str, str]:
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}


def _unit(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


if __name__ == "__main__":
    main()
