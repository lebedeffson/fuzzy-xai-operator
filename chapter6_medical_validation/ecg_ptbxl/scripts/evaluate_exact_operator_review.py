"""Evaluate the identical frozen ECG system operator without rendering per-record reports.

The canonical source provider and system builder are the very functions called
by runtime.py. Native IG remains the separate, already executed selected-case
experiment. Compare with archived full public routes and fresh lightweight
public routes before scoring the cohort. No change to training, risk weights,
rank comparator, review budget or held-out labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from fuzzyxai.adapters.model import ModelPrediction
from fuzzyxai.adapters.system_source import derive_system_source_evidence
from fuzzyxai.system_semantics import build_system_evidence

from chapter6_medical_validation.ecg_ptbxl.src.data import LEADS
from chapter6_medical_validation.ecg_ptbxl.src.model import build_ecg_resnet1d
from chapter6_medical_validation.ecg_ptbxl.src.preprocessing import normalize
from chapter6_medical_validation.ecg_ptbxl.scripts.evaluate_retrain_independent import (
    RouteOnlyAdapter, assess_review, dump, quality
)
from chapter6_medical_validation.shared.fuzzyxai_adapter import (
    SystemRunFacts, explain_public, load_plan, observation_context
)
from chapter6_medical_validation.shared.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def core_route(facts, prediction, plan, fingerprint):
    """Exactly the runtime.py source-provider -> build_system_evidence chain."""
    context = observation_context(facts, plan)
    obs = context.system_observation
    source = derive_system_source_evidence(
        object_id=facts.object_id,
        model_fingerprint=fingerprint,
        prediction=prediction,
        internal_evidence={},  # Torch binary class probability provider, no ensemble votes.
        source_interface_id=obs.alignment_transform.source_interface,
        risk_class=obs.risk_class,
        trace=obs.trace,
        model_trace=obs.model_trace,
        source_refs=obs.source_refs,
    )
    return build_system_evidence(
        object_id=facts.object_id, model_fingerprint=fingerprint,
        source=source, plan=plan, observation=obs
    )


def compare_system(system, rho, gamma, action, critical):
    dr = abs(float(system.risk.rho) - float(rho))
    dg = abs(float(system.alignment["gamma"]) - float(gamma))
    assert dr < 1e-9 and dg < 1e-9, (dr, dg)
    assert system.risk.action == action
    assert bool(system.risk.critical_override) == bool(critical)
    return {"abs_rho_difference": dr, "abs_gamma_difference": dg,
            "same_action": True, "same_critical_override": True}


def main():
    parser = argparse.ArgumentParser()
    for flag in ("data-root", "runs", "archive", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(int(os.environ.get("ECG_NUM_THREADS", "4")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = args.data_root / "ecg" / "ptb-xl-1.0.3" / "prepared"
    manifest_path = prepared / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["prepared_sha256"].items():
        assert sha256_file(prepared / name) == digest, name
    assert manifest["all_43598_records100_files_verified"]
    signals = np.load(prepared / "signals.npy", mmap_mode="r", allow_pickle=False)
    labels, folds, ids = (
        np.load(prepared / name, allow_pickle=False)
        for name in ("labels.npy", "folds.npy", "ecg_ids.npy")
    )
    refs = json.loads((prepared / "source_pairs.json").read_text())
    stats = json.loads((prepared / "normalization.json").read_text())
    table = pd.read_csv(prepared / "label_construction.csv").set_index("ecg_id")
    runs = [(p.parent, json.loads(p.read_text())) for p in sorted(args.runs.rglob("run.json"))]
    assert len(runs) == 3 and {v["seed"] for _, v in runs} == {2026, 2027, 2028}
    for path, run in runs:
        assert sha256_file(path / "best.pt") == run["checkpoint_sha256"]
        assert run["execution"]["dataset_manifest_sha256"] == sha256_file(manifest_path)
    run_dir, run = min(runs, key=lambda pair: (
        min(x["validation_loss"] for x in pair[1]["history"]),
        -max(x["validation_auroc"] for x in pair[1]["history"]), pair[1]["seed"]
    ))
    cfg = yaml.safe_load((ROOT / "configs" / "model_ecg_resnet1d.yaml").read_text())
    model = build_ecg_resnet1d(tuple(cfg["channels"]), int(cfg["blocks_per_stage"])).to(device).eval()
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True)["state_dict"])
    temp = float(run["calibration"]["temperature"])
    plan_path = ROOT / "configs" / "explain_plan_ecg.yaml"
    plan = load_plan(plan_path)
    normal_sha = sha256_file(prepared / "normalization.json")
    source_archive = json.loads((args.archive / "independent_xai_summary.json").read_text())
    saved_cases = source_archive["cases"]
    assert len(saved_cases) == 11
    fingerprints = {
        json.loads((args.archive / f"ptbxl-{x['ecg_id']}" / "public_result.json").read_text())
        ["provenance"]["model_fingerprint"] for x in saved_cases
    }
    assert len(fingerprints) == 1
    fingerprint = next(iter(fingerprints))

    def transform(value):
        return torch.from_numpy(normalize(np.asarray(value, dtype=np.float32).reshape(12, 1000), stats))[None].to(device)
    adapter = RouteOnlyAdapter(model, temperature=temp, task="classification", ig_steps=64, input_transform=transform)
    feature_names = [f"{lead}_t{t:04d}" for lead in LEADS for t in range(1000)]
    test = np.load(run_dir / "test_predictions.npz", allow_pickle=False)
    indices = test["prepared_indices"]
    assert len(indices) == 2107 and np.array_equal(indices, np.flatnonzero(folds == 10))
    assert np.array_equal(test["labels"], labels[indices])
    cached_probs = torch.softmax(torch.from_numpy(test["logits"]) / temp, 1).numpy()
    index_by_id = {int(ids[i]): int(i) for i in indices}

    def facts_for(ecg_id, prediction):
        ref = refs[str(ecg_id)]
        patient_hash = hashlib.sha256(f"ptbxl:{int(table.loc[ecg_id,'patient_id'])}".encode()).hexdigest()
        probs = tuple(float(v) for v in prediction.probabilities[0])
        predicted = int(np.argmax(probs))
        return SystemRunFacts(
            "ptb-xl-1.0.3-records100", "test", f"ptbxl-{ecg_id}", ref["source_sha256"],
            {"version": "ptbxl_train_lead_zscore_v1", "lead_order": list(LEADS),
             "sample_rate_hz": 100, "normalization_sha256": normal_sha},
            "ECGResNet1D", run["checkpoint_sha256"], run["run_id"], probs, predicted, 1,
            (patient_hash, *ref["files"].values())
        ), patient_hash

    # Use exactly the original probability inputs from 11 previously executed
    # full public routes to check arithmetic and action equality, not merely rank.
    archived_checks = []
    for case in saved_cases:
        ecg_id = case["ecg_id"]
        assert case["checkpoint_sha256"] == run["checkpoint_sha256"]
        assert case["source_sha256"] == refs[str(ecg_id)]["source_sha256"]
        public = json.loads((args.archive / f"ptbxl-{ecg_id}" / "public_result.json").read_text())
        pred = ModelPrediction(
            predictions=public["prediction"]["value"],
            probabilities=public["prediction"]["probabilities"],
            model_type=public["provenance"]["model_type"],
            adapter_id=public["provenance"]["adapter_id"],
        )
        facts, _ = facts_for(ecg_id, pred)
        system = core_route(facts, pred, plan, fingerprint)
        archived_checks.append({"ecg_id": ecg_id, **compare_system(
            system, case["rho"], case["gamma"], case["action"], case["critical_override"]
        )})
    # Fresh full lightweight routes on actual signals are a second cross-check.
    fresh_checks = []
    for ecg_id in (180, 504, 4079, 4362):
        raw = np.asarray(signals[index_by_id[ecg_id]], dtype=np.float32)
        numeric = raw.reshape(1, -1)
        pred = adapter.predict(numeric)
        facts, _ = facts_for(ecg_id, pred)
        system = core_route(facts, pred, plan, fingerprint)
        q_evidence, _ = quality(raw)
        public = explain_public(model, adapter, numeric, raw, facts, plan_path=plan_path,
                                attribution_maps=[], technical_evidence=q_evidence,
                                feature_names=feature_names)
        assert public.to_dict(detail="standard")["provenance"]["model_fingerprint"] == fingerprint
        native = public.system
        fresh_checks.append({"ecg_id":ecg_id, **compare_system(
            system, native.risk.rho, native.alignment["gamma"],
            native.risk.action, native.risk.critical_override
        )})
    equivalence = {"archive_full_routes": archived_checks, "fresh_lightweight_public_routes": fresh_checks,
                   "exact_same_core_functions": True, "frozen_model_fingerprint": fingerprint,
                   "source_run_id": 45779528570, "native_attributions": "selected cases separately evaluated"}
    dump(args.out / "equivalence_checks.json", equivalence)
    print("EQUIVALENCE PASSED", len(archived_checks), len(fresh_checks), flush=True)

    # New model forward on every verified raw record, then use the canonical
    # system operator directly. No test reference label passed to the route.
    rows = []
    max_difference = 0.0
    prediction_changes = 0
    for position, idx in enumerate(indices):
        index, ecg_id = int(idx), int(ids[idx])
        raw = np.asarray(signals[index], dtype=np.float32)
        pred = adapter.predict(raw.reshape(1, -1))
        facts, patient_hash = facts_for(ecg_id, pred)
        _, q = quality(raw)
        system = core_route(facts, pred, plan, fingerprint)
        risk = system.risk
        assert risk.rho is not None and np.isfinite(risk.rho)
        diff = np.abs(np.asarray(facts.probabilities)-cached_probs[position]).max()
        max_difference = max(max_difference, float(diff))
        prediction_changes += int(facts.predicted_class != int(np.argmax(cached_probs[position])))
        row = {"ecg_id": ecg_id, "prepared_index": index, "patient_hash": patient_hash,
               "prediction": facts.predicted_class, "p_abnormal": facts.probabilities[1],
               "technical_quality_score": q, "rho": float(risk.rho), "action": risk.action,
               "critical_override": bool(risk.critical_override),
               "gamma": float(system.alignment["gamma"]),
               "checkpoint_sha256": run["checkpoint_sha256"], "source_sha256": facts.source_sha256}
        rows.append(row)
        with (args.out / "records.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False)+"\n")
        if len(rows) % 100 == 0:
            print("CORE REVIEW", len(rows), flush=True)
    assert len(rows) == 2107
    assert max_difference < .0001, max_difference
    for row in rows:
        row["truth"] = int(labels[row["prepared_index"]])
    report = assess_review(rows, budget=.20, bootstrap=2000)
    report["execution"] = {
        "reference_run_id": 45779528570, "canonical_seed": run["seed"],
        "checkpoint_sha256": run["checkpoint_sha256"], "dataset_manifest_sha256": sha256_file(manifest_path),
        "implementation": "canonical derive_system_source_evidence -> build_system_evidence; no report construction",
        "equivalence": equivalence, "max_abs_diff_vs_cached_probabilities": max_difference,
        "predictions_changed_vs_cached": prediction_changes, "device": str(device)
    }
    dump(args.out / "retrospective_review.json", report)
    dump(args.out / "test_rows_with_reference_labels.json", rows)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("COMPLETE EXACT CORE REVIEW", len(rows), flush=True)


if __name__ == "__main__":
    main()
