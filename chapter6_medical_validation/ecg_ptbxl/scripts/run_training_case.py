"""Export one same-run ECG training trajectory through the public API.

The fixed validation probe was measured after every epoch by ``train_ecg``.
This script explains that *same* final-checkpoint model and registers the
measured trajectory with ``FuzzyXAI.observe_training`` before ``explain_one``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from chapter6_medical_validation.ecg_ptbxl.scripts.run_cases import canonical_run
from chapter6_medical_validation.ecg_ptbxl.src.data import LEADS
from chapter6_medical_validation.ecg_ptbxl.src.model import build_ecg_resnet1d
from chapter6_medical_validation.ecg_ptbxl.src.preprocessing import normalize, technical_signal_quality
from chapter6_medical_validation.ecg_ptbxl.src.xai import temporal_occlusion
from chapter6_medical_validation.shared.fuzzyxai_adapter import SystemRunFacts, explain_public
from chapter6_medical_validation.shared.hashing import sha256_file, sha256_json
from chapter6_medical_validation.shared.torch_adapter import TemperatureScaledTorchAdapter

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data_root = os.environ.get("FUZZYXAI_CH6_DATA_ROOT")
    if not data_root:
        raise FileNotFoundError("FUZZYXAI_CH6_DATA_ROOT is not set")
    prepared = Path(data_root) / "ecg" / "ptb-xl-1.0.3" / "prepared"
    signals = np.load(prepared / "signals.npy", mmap_mode="r")
    construction = pd.read_csv(prepared / "label_construction.csv")
    stats = json.loads((prepared / "normalization.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "configs" / "model_ecg_resnet1d.yaml").read_text(encoding="utf-8"))
    run_dir, run = canonical_run()
    probe = dict(run["training_probe"])
    index, ecg_id = int(probe["prepared_index"]), int(probe["ecg_id"])
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model = build_ecg_resnet1d(tuple(config["channels"]), int(config["blocks_per_stage"])).cuda().eval()
    model.load_state_dict(checkpoint["state_dict"])
    temperature = float(run["calibration"]["temperature"])
    raw = np.asarray(signals[index], dtype=np.float32)

    def input_transform(value: object, *, statistics: dict[str, object] = stats) -> torch.Tensor:
        source = np.asarray(value, dtype=np.float32).reshape(12, 1000)
        return torch.as_tensor(normalize(source, statistics), dtype=torch.float32, device="cuda")[None]

    adapter = TemperatureScaledTorchAdapter(model, temperature=temperature, task="classification", ig_steps=64, input_transform=input_transform)
    numeric_input = raw.reshape(1, -1)
    prediction = adapter.predict(numeric_input)
    probabilities = tuple(float(value) for value in prediction.probabilities[0])
    predicted = int(np.argmax(probabilities))
    standardized = input_transform(raw)
    occlusion = temporal_occlusion(model, standardized, target=predicted, temperature=temperature, window=50, stride=50)
    occlusion_full = np.repeat(np.asarray(occlusion["importance"], dtype=np.float32), 50, axis=1)
    source_row = construction.loc[construction["ecg_id"] == ecg_id].iloc[0]
    filename = str(source_row["filename_lr"])
    waveform_refs = {
        suffix: sha256_file(Path(data_root) / "ecg" / "ptb-xl-1.0.3" / f"{filename}{suffix}")
        for suffix in (".dat", ".hea")
    }
    patient_hash = hashlib.sha256(f"ptbxl:{int(source_row['patient_id'])}".encode()).hexdigest()
    facts = SystemRunFacts(
        "ptb-xl-1.0.3-records100", "validation", f"ptbxl-{ecg_id}", sha256_json(waveform_refs),
        {"version": "ptbxl_train_lead_zscore_v1", "lead_order": list(LEADS), "sample_rate_hz": 100,
         "normalization_sha256": sha256_file(prepared / "normalization.json")},
        "ECGResNet1D", sha256_file(checkpoint_path), str(run["run_id"]), probabilities, predicted, 1,
        (patient_hash, *waveform_refs.values()),
    )
    from fuzzyxai import FuzzyXAI

    training_history = {"objects": {facts.object_id: list(probe["history_through_final_checkpoint"])}}
    training_analysis = FuzzyXAI.wrap(model, adapter=adapter).observe_training(
        history=training_history,
        training_run_id=str(run["run_id"]),
        training_method="AdamW ECGResNet1D; per-epoch measured fixed validation probe",
        epoch_source=str(probe["epoch_source"]),
        final_checkpoint_ref=f"{checkpoint_path.name}:epoch:{int(checkpoint['epoch'])};sha256:{sha256_file(checkpoint_path)}",
    )
    result = explain_public(
        model, adapter, numeric_input, raw, facts,
        plan_path=ROOT / "configs" / "explain_plan_ecg.yaml",
        attribution_maps=[{
            "method": "temporal_occlusion", "values": occlusion_full, "target": predicted,
            "baseline": "zero standardized train-mean", "source_refs": ["temporal_occlusion_50_sample_windows"],
        }],
        technical_evidence=technical_signal_quality(raw), training_run=training_analysis,
        feature_names=[f"{lead}_t{sample:04d}" for lead in LEADS for sample in range(1000)],
    )
    output = ROOT / "outputs" / "training" / facts.object_id
    output.mkdir(parents=True, exist_ok=True)
    result.export_json(output / "result.json", detail="standard")
    (output / "audit.json").write_text(json.dumps(result.audit(), indent=2) + "\n", encoding="utf-8")
    (output / "reader_ru.txt").write_text(result.full_report(level="reader"), encoding="utf-8")
    (output / "audit_ru.txt").write_text(result.full_report(level="audit"), encoding="utf-8")
    (output / "provenance_action.json").write_text(json.dumps(result.inspect("action").to_dict(), indent=2) + "\n", encoding="utf-8")
    result.visualize(view="provenance", selector="action", output=output / "provenance_action.png")
    trace = training_analysis.traces[facts.object_id]
    (output / "training_summary.json").write_text(json.dumps({
        "training_run_id": trace.training_run_id, "model_fingerprint": trace.model_fingerprint,
        "checkpoint_sha256": facts.checkpoint_sha256, "final_checkpoint_ref": trace.final_checkpoint_ref,
        "training_method": trace.training_method, "epoch_source": trace.epoch_source,
        "first_learned_epoch": trace.first_learned_epoch, "forgetting_events": list(trace.forgetting_events),
        "forgetting_details": list(trace.forgetting_details), "stability": trace.stability_score,
        "loss_status": trace.loss_status, "history": list(trace.epoch_metrics),
    }, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
