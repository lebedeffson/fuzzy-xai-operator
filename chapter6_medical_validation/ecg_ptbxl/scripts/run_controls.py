"""Controlled integrity-fault checks for the real PTB-XL public route.

The waveform, final checkpoint and public explanation route are unchanged.
Only registered factual provenance is fault-injected, so a block demonstrates
the fail-closed system policy rather than an asserted medical error.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from chapter6_medical_validation.ecg_ptbxl.scripts.run_cases import canonical_run
from chapter6_medical_validation.ecg_ptbxl.src.data import LEADS
from chapter6_medical_validation.ecg_ptbxl.src.model import build_ecg_resnet1d
from chapter6_medical_validation.ecg_ptbxl.src.preprocessing import normalize, technical_signal_quality
from chapter6_medical_validation.shared.fuzzyxai_adapter import SystemRunFacts, explain_public
from chapter6_medical_validation.shared.hashing import sha256_file, sha256_json
from chapter6_medical_validation.shared.torch_adapter import TemperatureScaledTorchAdapter

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data_root = os.environ.get("FUZZYXAI_CH6_DATA_ROOT")
    if not data_root:
        raise FileNotFoundError("FUZZYXAI_CH6_DATA_ROOT is not set")
    prepared = Path(data_root) / "ecg" / "ptb-xl-1.0.3" / "prepared"
    selected = json.loads((ROOT / "outputs" / "cases" / "selected_cases.json").read_text(encoding="utf-8"))["cases"]["ECG_A"]
    index, ecg_id = int(selected["prepared_index"]), int(selected["ecg_id"])
    signals = np.load(prepared / "signals.npy", mmap_mode="r")
    construction = pd.read_csv(prepared / "label_construction.csv")
    stats = json.loads((prepared / "normalization.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "configs" / "model_ecg_resnet1d.yaml").read_text(encoding="utf-8"))
    run_dir, run = canonical_run()
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model = build_ecg_resnet1d(tuple(config["channels"]), int(config["blocks_per_stage"])).cuda().eval()
    model.load_state_dict(checkpoint["state_dict"])
    raw = np.asarray(signals[index], dtype=np.float32)

    def transform(value: object, *, statistics: dict[str, object] = stats) -> torch.Tensor:
        signal = np.asarray(value, dtype=np.float32).reshape(12, 1000)
        return torch.as_tensor(normalize(signal, statistics), dtype=torch.float32, device="cuda")[None]

    adapter = TemperatureScaledTorchAdapter(model, temperature=float(run["calibration"]["temperature"]), task="classification", ig_steps=64, input_transform=transform)
    numeric_input = raw.reshape(1, -1)
    prediction = adapter.predict(numeric_input)
    probabilities = tuple(float(value) for value in prediction.probabilities[0])
    predicted = int(np.argmax(probabilities))
    source_row = construction.loc[construction["ecg_id"] == ecg_id].iloc[0]
    filename = str(source_row["filename_lr"])
    refs = {suffix: sha256_file(Path(data_root) / "ecg" / "ptb-xl-1.0.3" / f"{filename}{suffix}") for suffix in (".dat", ".hea")}
    facts = SystemRunFacts(
        "ptb-xl-1.0.3-records100", "test", f"ptbxl-{ecg_id}", sha256_json(refs),
        {"version": "ptbxl_train_lead_zscore_v1", "lead_order": list(LEADS), "sample_rate_hz": 100,
         "normalization_sha256": sha256_file(prepared / "normalization.json")},
        "ECGResNet1D", sha256_file(checkpoint_path), str(run["run_id"]), probabilities, predicted, 1,
        (hashlib.sha256(f"ptbxl:{int(source_row['patient_id'])}".encode()).hexdigest(), *refs.values()),
    )
    ig = np.load(ROOT / "outputs" / "cases" / facts.object_id / "integrated_gradients_signed.npy").astype(np.float32)
    faults = {
        "ECG_H_missing_provenance": (facts, {"critical": True, "code": "controlled_missing_waveform_provenance", "source_refs": [facts.object_id, "waveform_hash_required"]}),
        "ECG_I_checkpoint_mismatch": (replace(facts, checkpoint_sha256="0" * 64), {"critical": True, "code": "controlled_checkpoint_mismatch", "source_refs": [facts.object_id, facts.checkpoint_sha256]}),
        "ECG_J_attribution_target_mismatch": (facts, {"critical": True, "code": "controlled_attribution_target_mismatch", "source_refs": [facts.object_id, f"prediction_target:{predicted}", f"registered_mismatch_target:{1-predicted}"]}),
    }
    output_root = ROOT / "outputs" / "controls"
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}
    for control_id, (control_facts, fault) in faults.items():
        result = explain_public(
            model, adapter, numeric_input, raw, control_facts,
            plan_path=ROOT / "configs" / "explain_plan_ecg.yaml",
            attribution_maps=[{"method": "integrated_gradients", "values": ig, "target": predicted,
                               "baseline": "zero standardized train-mean", "source_refs": ["reused_normal_case_IG_for_integrity_control"]}],
            technical_evidence=technical_signal_quality(raw), critical_fault=fault,
            feature_names=[f"{lead}_t{sample:04d}" for lead in LEADS for sample in range(1000)],
        )
        target = output_root / control_id
        target.mkdir(exist_ok=False)
        result.export_json(target / "result.json", detail="standard")
        (target / "audit.json").write_text(json.dumps(result.audit(), indent=2) + "\n", encoding="utf-8")
        (target / "reader_ru.txt").write_text(result.full_report(level="reader"), encoding="utf-8")
        result.visualize(view="provenance", selector="action", output=target / "provenance_action.png")
        system = result.system
        summary[control_id] = {"rho": system.risk.rho, "candidate_action": system.risk.candidate_action,
                               "critical_override": system.risk.critical_override, "final_action": system.risk.action,
                               "diagnostics": list(system.diagnostics)}
    (output_root / "controls_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(output_root)


if __name__ == "__main__":
    main()
