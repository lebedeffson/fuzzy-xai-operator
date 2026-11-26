"""Independent raw-waveform attribution and exploratory equal-budget review."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from fuzzyxai.adapters.contracts_v2 import LocalModelEvidence

from chapter6_medical_validation.ecg_ptbxl.src.data import LEADS
from chapter6_medical_validation.ecg_ptbxl.src.model import build_ecg_resnet1d
from chapter6_medical_validation.ecg_ptbxl.src.preprocessing import normalize, technical_signal_quality
from chapter6_medical_validation.ecg_ptbxl.scripts.run_cases import select_initial_cases, faithfulness
from chapter6_medical_validation.shared.fuzzyxai_adapter import SystemRunFacts, explain_public
from chapter6_medical_validation.shared.torch_adapter import TemperatureScaledTorchAdapter
from chapter6_medical_validation.shared.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
FIXED_IDS = (180, 504, 4079, 7895, 14771, 16534)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def independent_ig(model, x, target, intervals=64, batch=16):
    """Independent trapezoid integration of a fixed, uncalibrated logit."""
    assert x.shape == (1, 12, 1000) and intervals > 0
    model.eval()
    alphas = torch.linspace(0, 1, intervals + 1, device=x.device, dtype=x.dtype)
    total = torch.zeros_like(x, dtype=torch.float64)
    for start in range(0, len(alphas), batch):
        values = alphas[start:start + batch]
        points = (x * values.view(-1, 1, 1)).detach().requires_grad_(True)
        grad, = torch.autograd.grad(model(points)[:, target].sum(), points)
        weights = torch.ones(len(values), dtype=torch.float64, device=x.device)
        if start == 0:
            weights[0] = .5
        if start + len(values) == len(alphas):
            weights[-1] = .5
        total += (grad.double() * weights.view(-1, 1, 1)).sum(0, keepdim=True)
    result = (total * x.double() / intervals)[0].detach().cpu().numpy()
    with torch.no_grad():
        difference = float((model(x)[0, target] - model(torch.zeros_like(x))[0, target]).item())
    return result, {"target": int(target), "intervals": intervals, "logit_difference": difference,
                    "sum_attributions": float(result.sum()), "completeness_residual": float(abs(result.sum() - difference))}


def independent_occlusion(model, x, target, temperature, batch=64):
    """Signed difference in calibrated target probability after one cell is masked."""
    with torch.no_grad():
        p = float(torch.softmax(model(x) / temperature, 1)[0, target])
        out = []
        for start in range(0, 240, batch):
            cells = list(range(start, min(start + batch, 240)))
            masked = x.repeat(len(cells), 1, 1)
            for row, cell in enumerate(cells):
                lead, t = divmod(cell, 20)
                masked[row, lead, t * 50:(t + 1) * 50] = 0
            out.extend((p - torch.softmax(model(masked) / temperature, 1)[:, target]).cpu().tolist())
    return np.asarray(out, dtype=float).reshape(12, 20)


def unit_l1(a):
    return a / np.abs(a).sum() if np.abs(a).sum() else a


class RouteOnlyAdapter(TemperatureScaledTorchAdapter):
    """The same public system route, with missing cohort-wide native maps explicit."""

    def extract_local_evidence(self, inputs, prediction, context):
        return LocalModelEvidence(
            channels={"contribution_method": "not_run_lightweight_system_route"},
            missing_channels=("native_attribution_not_repeated_for_aggregate_route",),
            limitations=("Native attribution evaluated separately for selected ECG records.",),
        )


def quality(raw):
    q = technical_signal_quality(raw)
    score = float(q["finite_fraction"]) * (1.0 - max(float(np.mean(q["flatline_fraction_per_lead"])), float(np.mean(q["extreme_fraction_per_lead"]))))
    return q, score


def rank_mask(score, ids, k):
    score, ids = np.asarray(score, dtype=float), np.asarray(ids, dtype=int)
    if score.shape != ids.shape or not np.isfinite(score).all():
        raise ValueError("Nonfinite/incomplete ranking score")
    order = np.lexsort((ids, -score))
    mask = np.zeros(len(ids), dtype=bool)
    mask[order[:k]] = True
    return mask


def assess_review(rows, budget=.20, bootstrap=2000):
    ids = np.array([x["ecg_id"] for x in rows])
    pred = np.array([x["prediction"] for x in rows])
    y = np.array([x["truth"] for x in rows])
    p = np.array([x["p_abnormal"] for x in rows])
    q = np.array([x["technical_quality_score"] for x in rows])
    rho = np.array([x["rho"] for x in rows])
    conf = np.maximum(p, 1-p)
    groups = np.array([x["patient_hash"] for x in rows])
    _, inv = np.unique(groups, return_inverse=True)
    n = len(rows)
    k = int(np.floor(n * budget))
    assert n == len(set(ids.tolist())) and 0 < k < n
    err = y != pred
    scores = {"fuzzyxai_lightweight_rho": rho, "confidence": 1-conf, "risk_class_probability": p,
              "confidence_or_quality": np.maximum(1-conf, 1-q)}
    masks = {name: rank_mask(s, ids, k) for name, s in scores.items()}
    metrics = {}
    for name, review in masks.items():
        accept = ~review
        metrics[name] = {"review_n": int(review.sum()), "accept_n": int(accept.sum()), "flagged_errors": int((review & err).sum()),
                         "remaining_errors": int((accept & err).sum()),
                         "remaining_FN": int((accept & (y == 1) & (pred == 0)).sum()),
                         "remaining_FP": int((accept & (y == 0) & (pred == 1)).sum()),
                         "accepted_error_rate": float((accept & err).sum()/accept.sum()),
                         "fraction_errors_flagged": float((review & err).sum()/err.sum()) if err.sum() else None,
                         "coverage": float(accept.mean())}
    # Paired patient-cluster bootstrap of the observed fixed policies, not
    # a new threshold fit or a simulated clinician.
    n_groups = int(inv.max()) + 1
    group_sizes = np.bincount(inv, minlength=n_groups)
    rng = np.random.default_rng(20261126)
    baseline = masks["fuzzyxai_lightweight_rho"]
    comparisons = {}
    for name, other in masks.items():
        if name == "fuzzyxai_lightweight_rho":
            continue
        delta = (baseline.astype(int) - other.astype(int)) * err
        counts = np.bincount(inv, weights=delta, minlength=n_groups)
        boot = []
        for _ in range(bootstrap):
            sample = rng.integers(n_groups, size=n_groups)
            boot.append(float(100*counts[sample].sum()/group_sizes[sample].sum()))
        comparisons[name] = {"additional_errors_flagged_per_100_records": float(100*delta.sum()/n),
                             "patient_cluster_bootstrap_CI95": [float(x) for x in np.quantile(boot, [.025,.975])]}
    return {"scope": "exploratory retrospective ranking; no clinician adjudication or clinical outcome claim",
            "test_n": n, "patients": n_groups, "review_budget": budget, "review_n_each": k,
            "total_errors": int(err.sum()), "metrics": metrics, "paired_comparisons": comparisons,
            "fixed_native_actions": {a: sum(x["action"] == a for x in rows) for a in sorted({x["action"] for x in rows})}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("xai", "review"), required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(int(os.environ.get("ECG_NUM_THREADS", "4")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = args.data_root / "ecg" / "ptb-xl-1.0.3" / "prepared"
    manifest = json.loads((prepared / "dataset_manifest.json").read_text())
    for name, expected in manifest["prepared_sha256"].items():
        assert sha256_file(prepared / name) == expected, name
    signals = np.load(prepared / "signals.npy", mmap_mode="r")
    labels, folds, ids = (np.load(prepared / name, allow_pickle=False) for name in ("labels.npy","folds.npy","ecg_ids.npy"))
    refs = json.loads((prepared / "source_pairs.json").read_text())
    stats = json.loads((prepared / "normalization.json").read_text())
    construction = pd.read_csv(prepared / "label_construction.csv").set_index("ecg_id")
    run_paths = sorted(args.runs.rglob("run.json"))
    runs = [(path.parent, json.loads(path.read_text())) for path in run_paths]
    assert {r["seed"] for _, r in runs} == {2026, 2027, 2028} and len(runs) == 3
    for path, run in runs:
        assert sha256_file(path / "best.pt") == run["checkpoint_sha256"], path
        assert run["execution"]["dataset_manifest_sha256"] == sha256_file(prepared / "dataset_manifest.json")
    run_dir, run = min(runs, key=lambda pair: (min(x["validation_loss"] for x in pair[1]["history"]), -max(x["validation_auroc"] for x in pair[1]["history"]), pair[1]["seed"]))
    cfg = yaml.safe_load((ROOT / "configs" / "model_ecg_resnet1d.yaml").read_text())
    model = build_ecg_resnet1d(tuple(cfg["channels"]), int(cfg["blocks_per_stage"])).to(device).eval()
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    temperature = float(run["calibration"]["temperature"])
    dump(args.out / "training_comparison.json", {"canonical_seed": run["seed"], "canonical_checkpoint_sha256": run["checkpoint_sha256"],
         "all_runs": [r for _, r in runs], "data_manifest": manifest, "software": {"torch": str(torch.__version__), "numpy": str(np.__version__), "device": str(device)}})
    test = np.load(run_dir / "test_predictions.npz", allow_pickle=False)
    test_indices = test["prepared_indices"]
    assert np.array_equal(test_indices, np.flatnonzero(folds == 10)) and len(test_indices) == 2107
    assert np.array_equal(test["labels"], labels[test_indices])
    def transform(value):
        return torch.from_numpy(normalize(np.asarray(value, dtype=np.float32).reshape(12,1000), stats))[None].to(device)
    adapter_class = TemperatureScaledTorchAdapter if args.mode == "xai" else RouteOnlyAdapter
    adapter = adapter_class(model, temperature=temperature, task="classification", ig_steps=64, input_transform=transform)
    plan = ROOT / "configs" / "explain_plan_ecg.yaml"
    feature_names = [f"{lead}_t{t:04d}" for lead in LEADS for t in range(1000)]
    results = []
    if args.mode == "xai":
        # Historical fixed cases plus deterministic descriptive cases from the
        # new frozen predictions; the latter are never used to score the review policy.
        probs = torch.softmax(torch.from_numpy(test["logits"]) / temperature, dim=1).numpy()
        rows = []
        for i, index in enumerate(test_indices):
            _, q = quality(np.asarray(signals[index]))
            pred = int(np.argmax(probs[i]))
            rows.append({"prepared_index": int(index), "ecg_id": int(ids[index]), "label": int(labels[index]),
                         "prediction": pred, "p_abnormal": float(probs[i,1]), "confidence": float(probs[i,pred]),
                         "correct": pred == int(labels[index]), "technical_quality_score": q})
        picked = select_initial_cases(rows)
        wanted = set(FIXED_IDS) | {int(x["ecg_id"]) for x in picked.values() if "ecg_id" in x}
        test_indices = [i for i in test_indices if int(ids[i]) in wanted]
        assert set(FIXED_IDS).issubset({int(ids[i]) for i in test_indices})
    normal_sha = sha256_file(prepared / "normalization.json")
    for idx in test_indices:
        index, ecg_id = int(idx), int(ids[idx])
        raw = np.asarray(signals[index], dtype=np.float32)
        numeric = raw.reshape(1,-1)
        x = transform(numeric)
        with torch.no_grad():
            logits = model(x)
            probabilities = tuple(float(v) for v in torch.softmax(logits / temperature,1)[0].cpu())
        target = int(np.argmax(probabilities))
        ref = refs[str(ecg_id)]
        patient_hash = hashlib.sha256(f"ptbxl:{int(construction.loc[ecg_id,'patient_id'])}".encode()).hexdigest()
        facts = SystemRunFacts("ptb-xl-1.0.3-records100", "test", f"ptbxl-{ecg_id}", ref["source_sha256"],
            {"version":"ptbxl_train_lead_zscore_v1","lead_order":list(LEADS),"sample_rate_hz":100,"normalization_sha256":normal_sha},
            "ECGResNet1D",run["checkpoint_sha256"],run["run_id"],probabilities,target,1,(patient_hash,*ref["files"].values()))
        q_evidence, q_score = quality(raw)
        maps = []
        if args.mode == "xai":
            ig, completeness = independent_ig(model, x, target)
            oc = independent_occlusion(model, x, target, temperature)
            assert np.isfinite(ig).all() and np.isfinite(oc).all()
            maps = [{"method":"temporal_occlusion","values":np.repeat(oc,50,axis=1),"target":target,
                     "baseline":"zero standardized train-mean","source_refs":["independent_50_sample_windows"]}]
        result = explain_public(model, adapter, numeric, raw, facts, plan_path=plan, attribution_maps=maps,
                                technical_evidence=q_evidence, feature_names=feature_names)
        system = result.system
        risk = system.risk
        row = {"ecg_id": ecg_id, "prepared_index": index, "patient_hash": patient_hash,
               "prediction": target, "p_abnormal": probabilities[1], "technical_quality_score": q_score,
               "rho": float(risk.rho) if risk.rho is not None else None, "action": risk.action,
               "critical_override": bool(risk.critical_override), "gamma": system.alignment["gamma"],
               "checkpoint_sha256": run["checkpoint_sha256"], "source_sha256": ref["source_sha256"]}
        if args.mode == "xai":
            case = args.out / f"ptbxl-{ecg_id}"
            case.mkdir()
            np.save(case/"integrated_gradients_independent.npy",ig,allow_pickle=False)
            np.save(case/"temporal_occlusion_independent.npy",oc,allow_pickle=False)
            native = result.to_dict(detail="audit", include_raw=True)["layers"]["attribution_maps"]
            native_ig = np.asarray(next(v["attribution_array"] for v in native if v["method"]=="integrated_gradients")).reshape(12,1000)
            np.save(case/"integrated_gradients_public_native.npy",native_ig,allow_pickle=False)
            a, b = unit_l1(ig.reshape(12,20,50).sum(2)), unit_l1(oc)
            corr = None if np.std(a)==0 or np.std(b)==0 else float(np.corrcoef(a.ravel(),b.ravel())[0,1])
            row.update({"truth":int(labels[index]),"independent_ig_completeness":completeness,
                        "native_vs_independent_ig_max_abs":float(np.abs(native_ig-ig).max()),
                        "native_vs_independent_ig_mean_abs":float(np.abs(native_ig-ig).mean()),
                        "D_XAI_diagnostic_not_system_Gamma":float(np.abs(a-b).sum()/2),
                        "signed_correlation":corr,"ig_masking":faithfulness(model,x,a,target=target,temperature=temperature,seed=2026+index),
                        "occlusion_masking":faithfulness(model,x,b,target=target,temperature=temperature,seed=3026+index),
                        "raw_signal_float32_sha256":hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
                        "official_wfdb_pair_checksums":ref["files"]})
            result.export_json(case/"public_result.json",detail="standard")
            dump(case/"independent_xai.json",row)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3,1,figsize=(10,7), constrained_layout=True)
            ax[0].plot(np.arange(1000)/100,raw[1],lw=.8);ax[0].set(xlim=(0,10),ylabel="Lead II (mV)",title=f"PTB-XL {ecg_id}; truth={labels[index]}, prediction={target}")
            for axis, values, title in ((ax[1],ig.reshape(12,20,50).sum(2),"Independent signed IG: logit"),(ax[2],oc,"Independent temporal occlusion: probability change")):
                m = max(float(np.abs(values).max()),1e-12)
                im=axis.imshow(values,aspect="auto",extent=(0,10,12,0),cmap="coolwarm",vmin=-m,vmax=m)
                axis.set(yticks=np.arange(12)+.5,yticklabels=LEADS,title=title,xlabel="Time (s)")
                fig.colorbar(im,ax=axis)
            fig.savefig(case/"waveform_and_attributions.png",dpi=160)
            plt.close(fig)
        results.append(row)
        with (args.out/"records.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(row,allow_nan=False)+"\n")
        if len(results) % 25 == 0 or args.mode == "xai":
            print(args.mode,len(results),ecg_id,flush=True)
    if args.mode == "review":
        # Reference labels were not passed to the public route; add only after scoring.
        assert len(results) == 2107 and all(x["rho"] is not None for x in results)
        for row in results:
            row["truth"] = int(labels[row["prepared_index"]])
        dump(args.out/"retrospective_review.json",assess_review(results))
        dump(args.out/"test_rows_with_reference_labels.json",results)
    else:
        dump(args.out/"independent_xai_summary.json",{"scope":"new checkpoint and original verified signals; not bitwise equivalence to unavailable prior model", "cases":results})
    print(f"COMPLETE {args.mode} {len(results)} records",flush=True)


if __name__ == "__main__":
    main()
