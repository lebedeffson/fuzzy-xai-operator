from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from chapter6_medical_validation.ecg_ptbxl.src.data import LEADS
from chapter6_medical_validation.ecg_ptbxl.src.model import build_ecg_resnet1d
from chapter6_medical_validation.ecg_ptbxl.src.preprocessing import normalize, technical_signal_quality
from chapter6_medical_validation.ecg_ptbxl.src.xai import common_ig_representation, common_occlusion_representation, temporal_occlusion
from chapter6_medical_validation.shared.fuzzyxai_adapter import SystemRunFacts, explain_public
from chapter6_medical_validation.shared.hashing import sha256_file, sha256_json
from chapter6_medical_validation.shared.torch_adapter import TemperatureScaledTorchAdapter

ROOT = Path(__file__).resolve().parents[1]


def canonical_run() -> tuple[Path, dict[str, object]]:
    candidates = []
    for path in (ROOT / "outputs" / "runs").glob("*/run.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        best_loss = min(float(row["validation_loss"]) for row in payload["history"])
        best_auroc = max(float(row["validation_auroc"]) for row in payload["history"])
        candidates.append((best_loss, -best_auroc, int(payload["seed"]), path.parent, payload))
    if not candidates:
        raise FileNotFoundError("no ECG training runs")
    _, _, _, run_dir, payload = min(candidates)
    return run_dir, payload


def select_initial_cases(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    groups = {
        "ECG_A": [row for row in rows if row["correct"] and row["label"] == 0],
        "ECG_B": [row for row in rows if row["correct"] and row["label"] == 1],
        "ECG_D": [row for row in rows if row["label"] == 0 and row["prediction"] == 1],
        "ECG_E": [row for row in rows if row["label"] == 1 and row["prediction"] == 0],
    }
    selected: dict[str, dict[str, object]] = {}
    for case_id, values in groups.items():
        selected[case_id] = max(values, key=lambda row: float(row["confidence"])) if values else {"status": "not_available", "reason": f"no case satisfying {case_id} policy"}
    selected["ECG_C"] = min(rows, key=lambda row: abs(float(row["p_abnormal"]) - 0.5))
    selected["ECG_G"] = min(rows, key=lambda row: float(row["technical_quality_score"]))
    return selected


def diagnostic_disagreement(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(np.asarray(left) - np.asarray(right)).sum() / 2.0)


def signed_correlation(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    """Return an explicit unavailable status instead of serializing NaN."""
    left_value = np.asarray(left, dtype=float).reshape(-1)
    right_value = np.asarray(right, dtype=float).reshape(-1)
    if np.std(left_value) == 0.0 or np.std(right_value) == 0.0:
        return {
            "value": None,
            "status": "not_applicable",
            "reason": "correlation is undefined because at least one common-grid attribution map is constant",
        }
    return {"value": float(np.corrcoef(left_value, right_value)[0, 1]), "status": "measured"}


def optional_metadata(value: object) -> object | None:
    return None if pd.isna(value) else value


def probability(model: torch.nn.Module, value: torch.Tensor, target: int, temperature: float) -> float:
    with torch.no_grad():
        return float(torch.softmax(model(value) / temperature, dim=1)[0, target].item())


def faithfulness(model: torch.nn.Module, standardized: torch.Tensor, importance: np.ndarray, *, target: int, temperature: float, seed: int) -> dict[str, object]:
    original = probability(model, standardized, target, temperature)
    magnitudes = np.abs(np.asarray(importance)).reshape(-1)
    count = max(1, int(np.ceil(0.10 * len(magnitudes))))

    def drop(indices: np.ndarray) -> float:
        masked = standardized.clone()
        for index in indices:
            lead, time_bin = divmod(int(index), 20)
            masked[0, lead, time_bin * 50 : (time_bin + 1) * 50] = 0.0
        return original - probability(model, masked, target, temperature)

    top = np.argsort(magnitudes)[-count:]
    rng = np.random.default_rng(seed)
    random_drops = [drop(rng.choice(len(magnitudes), size=count, replace=False)) for _ in range(20)]
    return {"target": target, "original_probability": original, "masked_fraction": 0.10, "masked_cells": count, "replacement": "zero standardized train-mean baseline", "top_drop": drop(top), "random_drop_mean": float(np.mean(random_drops)), "random_drop_std": float(np.std(random_drops)), "random_repeats": 20}


def main() -> None:
    data_root = os.environ.get("FUZZYXAI_CH6_DATA_ROOT")
    if not data_root:
        raise FileNotFoundError("FUZZYXAI_CH6_DATA_ROOT is not set")
    prepared = Path(data_root) / "ecg" / "ptb-xl-1.0.3" / "prepared"
    signals = np.load(prepared / "signals.npy", mmap_mode="r")
    labels = np.load(prepared / "labels.npy")
    ecg_ids = np.load(prepared / "ecg_ids.npy")
    construction = pd.read_csv(prepared / "label_construction.csv")
    database = pd.read_csv(Path(data_root) / "ecg" / "ptb-xl-1.0.3" / "ptbxl_database.csv", index_col="ecg_id")
    stats = json.loads((prepared / "normalization.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "configs" / "model_ecg_resnet1d.yaml").read_text(encoding="utf-8"))
    run_dir, run = canonical_run()
    test = np.load(run_dir / "test_predictions.npz")
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model = build_ecg_resnet1d(tuple(config["channels"]), int(config["blocks_per_stage"])).cuda().eval()
    model.load_state_dict(checkpoint["state_dict"])
    temperature = float(run["calibration"]["temperature"])
    calibrated = torch.softmax(torch.as_tensor(test["logits"]) / temperature, dim=1).numpy()
    summaries = []
    for position, prepared_index in enumerate(test["prepared_indices"]):
        index = int(prepared_index); quality = technical_signal_quality(np.asarray(signals[index])); predicted = int(np.argmax(calibrated[position])); confidence = float(calibrated[position, predicted])
        quality_score = float(quality["finite_fraction"]) * (1.0 - max(float(np.mean(quality["flatline_fraction_per_lead"])), float(np.mean(quality["extreme_fraction_per_lead"]))))
        summaries.append({"prepared_index": index, "ecg_id": int(ecg_ids[index]), "label": int(labels[index]), "prediction": predicted, "p_abnormal": float(calibrated[position, 1]), "confidence": confidence, "correct": predicted == int(labels[index]), "technical_quality_score": quality_score})
    selected = select_initial_cases(summaries)
    output_root = ROOT / "outputs" / "cases"; output_root.mkdir(parents=True, exist_ok=True)
    evaluated: dict[str, dict[str, object]] = {}
    feature_names = [f"{lead}_t{sample:04d}" for lead in LEADS for sample in range(1000)]
    unique_rows = {int(row["prepared_index"]): row for row in selected.values() if "prepared_index" in row}
    for index, row in unique_rows.items():
        raw = np.asarray(signals[index], dtype=np.float32); standardized_np = normalize(raw, stats); standardized = torch.as_tensor(standardized_np, dtype=torch.float32, device="cuda")[None]
        def input_transform(value: object, *, stats: dict[str, object] = stats) -> torch.Tensor:
            source = np.asarray(value, dtype=np.float32).reshape(12, 1000)
            return torch.as_tensor(normalize(source, stats), dtype=torch.float32, device="cuda")[None]
        adapter = TemperatureScaledTorchAdapter(model, temperature=temperature, task="classification", ig_steps=64, input_transform=input_transform)
        numeric_input = raw.reshape(1, -1); prediction = adapter.predict(numeric_input); probabilities = tuple(float(value) for value in prediction.probabilities[0]); predicted = int(np.argmax(probabilities))
        occlusion = temporal_occlusion(model, standardized, target=predicted, temperature=temperature, window=50, stride=50)
        occlusion_raw = np.asarray(occlusion["importance"], dtype=np.float32); occlusion_full = np.repeat(occlusion_raw, 50, axis=1)
        source_row = construction.loc[construction["ecg_id"] == int(ecg_ids[index])].iloc[0]; filename = str(source_row["filename_lr"]); waveform_refs = {suffix: sha256_file(Path(data_root) / "ecg" / "ptb-xl-1.0.3" / f"{filename}{suffix}") for suffix in (".dat", ".hea")}; source_hash = sha256_json(waveform_refs)
        patient_hash = hashlib.sha256(f"ptbxl:{int(source_row['patient_id'])}".encode()).hexdigest()
        facts = SystemRunFacts("ptb-xl-1.0.3-records100", "test", f"ptbxl-{int(ecg_ids[index])}", source_hash, {"version": "ptbxl_train_lead_zscore_v1", "lead_order": list(LEADS), "sample_rate_hz": 100, "normalization_sha256": sha256_file(prepared / "normalization.json")}, "ECGResNet1D", sha256_file(checkpoint_path), str(run["run_id"]), probabilities, predicted, 1, (patient_hash, *waveform_refs.values()))
        metadata_row = database.loc[int(ecg_ids[index])]
        quality = technical_signal_quality(raw); quality.update({name: optional_metadata(metadata_row.get(name)) for name in ("static_noise", "burst_noise", "baseline_drift", "electrodes_problems", "pacemaker")})
        result = explain_public(model, adapter, numeric_input, raw, facts, plan_path=ROOT / "configs" / "explain_plan_ecg.yaml", attribution_maps=[{"method": "temporal_occlusion", "values": occlusion_full, "target": predicted, "baseline": "zero standardized train-mean", "source_refs": ["temporal_occlusion_50_sample_windows"]}], technical_evidence=quality, feature_names=feature_names)
        case_dir = output_root / facts.object_id; case_dir.mkdir(exist_ok=False); result.export_json(case_dir / "result.json", detail="standard"); (case_dir / "audit.json").write_text(json.dumps(result.audit(), indent=2) + "\n"); (case_dir / "reader_ru.txt").write_text(result.full_report(level="reader")); (case_dir / "audit_ru.txt").write_text(result.full_report(level="audit")); (case_dir / "provenance_action.json").write_text(json.dumps(result.inspect("action").to_dict(), indent=2) + "\n"); result.visualize(view="provenance", selector="action", output=case_dir / "provenance_action.png")
        raw_payload = result.to_dict(detail="audit", include_raw=True); ig_entry = next(value for value in raw_payload["layers"]["attribution_maps"] if value["method"] == "integrated_gradients"); ig = np.asarray(ig_entry["attribution_array"], dtype=np.float32).reshape(12, 1000); np.save(case_dir / "integrated_gradients_signed.npy", ig); np.save(case_dir / "temporal_occlusion_signed.npy", occlusion_raw)
        ig_common = common_ig_representation(ig); occ_common = common_occlusion_representation(occlusion_raw); disagreement = diagnostic_disagreement(ig_common, occ_common)
        diagnostics = {"semantics": "registered common-grid diagnostic; not canonical Gamma", "T_IG": "signed 12x1000 -> sums in 12x20 -> signed L1 normalization", "T_OCC": "signed 12x20 -> signed L1 normalization", "normalized_l1_disagreement": disagreement, "signed_correlation": signed_correlation(ig_common, occ_common), "ig_faithfulness": faithfulness(model, standardized, ig_common, target=predicted, temperature=temperature, seed=2026 + index), "occlusion_faithfulness": faithfulness(model, standardized, occ_common, target=predicted, temperature=temperature, seed=3026 + index)}; (case_dir / "xai_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        system = result.system; evaluated[facts.object_id] = {**row, "object_id": facts.object_id, "probabilities": probabilities, "xai_diagnostic_disagreement": disagreement, "gamma": system.alignment["gamma"], "u_M": system.uncertainty.u_m, "Delta_status": system.reduction_status, "i_pre": system.i_pre, "rho": system.risk.rho, "candidate_action": system.risk.candidate_action, "critical_override": system.risk.critical_override, "action": system.risk.action}
    selected["ECG_F"] = max(evaluated.values(), key=lambda row: float(row["xai_diagnostic_disagreement"])) if evaluated else {"status": "not_available", "reason": "no XAI-evaluated natural case"}
    for case_id, row in list(selected.items()):
        if "ecg_id" in row:
            selected[case_id] = evaluated[f"ptbxl-{int(row['ecg_id'])}"]
    payload = {"selection_policy": "deterministic over frozen canonical test predictions; ECG_F is maximum among the predefined natural selected-case pool", "cases": selected}
    (output_root / "selected_cases.json").write_text(json.dumps(payload, indent=2) + "\n"); (output_root / "case_summaries.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(output_root)


if __name__ == "__main__":
    main()
