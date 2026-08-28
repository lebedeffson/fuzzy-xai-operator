from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from chapter6_medical_validation.brain_allen.scripts.run_cases import ROOT, canonical_run
from chapter6_medical_validation.brain_allen.src.model import build_inception_binary
from chapter6_medical_validation.brain_allen.src.preprocessing import preprocess_patch
from chapter6_medical_validation.shared.fuzzyxai_adapter import SystemRunFacts, explain_public
from chapter6_medical_validation.shared.hashing import sha256_file
from chapter6_medical_validation.shared.torch_adapter import TemperatureScaledTorchAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-name", default="prepared")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--protocol-id", default="brain_v1_pilot")
    args = parser.parse_args()
    data_root = os.environ.get("FUZZYXAI_CH6_DATA_ROOT")
    if not data_root:
        raise FileNotFoundError("FUZZYXAI_CH6_DATA_ROOT is not set")
    prepared = Path(data_root) / "brain" / "allen_ccf_25um" / args.prepared_name
    patches = np.load(prepared / "patches.npy", mmap_mode="r")
    metadata = json.loads((prepared / "patches.json").read_text(encoding="utf-8"))
    selected = json.loads((ROOT / args.output_root / "cases" / "selected_cases.json").read_text(encoding="utf-8"))["cases"]["BRAIN_A"]
    index = int(selected["prepared_index"]); item = metadata[index]
    run_dir, run = canonical_run(args.output_root); checkpoint_path = run_dir / "best.pt"; checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    # Match the constructor used during training; this flag controls a
    # torchvision input-transform behavior not carried by ``state_dict``.
    model = build_inception_binary(pretrained=True).cuda().eval(); model.load_state_dict(checkpoint["state_dict"]); scale = float(run["preprocessing"]["scale"])
    _, display, trace = preprocess_patch(np.asarray(patches[index]), scale=scale, split="test", seed=int(run["seed"]), object_index=index); numeric_input = np.asarray(patches[index], dtype=np.float32).reshape(1, -1)
    def transform(value: object) -> torch.Tensor:
        tensor, _, _ = preprocess_patch(np.asarray(value).reshape(patches.shape[1:]), scale=scale, split="test", seed=int(run["seed"]), object_index=index)
        return tensor[None].cuda()
    adapter = TemperatureScaledTorchAdapter(model, temperature=float(run["calibration"]["temperature"]), task="classification", ig_steps=64, input_transform=transform); prediction = adapter.predict(numeric_input); probabilities = tuple(float(value) for value in prediction.probabilities[0]); predicted = int(np.argmax(probabilities)); object_id = str(selected["object_id"])
    facts = SystemRunFacts(f"allen-ccf-2017-25um:{args.protocol_id}", "test", object_id, item["hash"], trace, "inception_v3", sha256_file(checkpoint_path), str(run["run_id"]), probabilities, predicted, 1, (item["hash"],))
    grad_cam = np.load(ROOT / args.output_root / "cases" / object_id / "grad_cam_raw.npy")
    controls = {
        "BRAIN_G_preprocessing_mismatch": (replace(facts, preprocessing_trace={**trace, "version": "controlled_wrong_preprocessing_version"}), {"critical": True, "code": "controlled_preprocessing_version_mismatch", "source_refs": [object_id, "expected:allen_nissl_preprocess_v1"]}),
        "BRAIN_H_checkpoint_mismatch": (replace(facts, checkpoint_sha256="0" * 64), {"critical": True, "code": "controlled_checkpoint_mismatch", "source_refs": [object_id, sha256_file(checkpoint_path)]}),
    }
    output_root = ROOT / args.output_root / "controls"; output_root.mkdir(parents=True, exist_ok=True); summary = {}
    for control_id, (control_facts, fault) in controls.items():
        result = explain_public(model, adapter, numeric_input, display, control_facts, plan_path=ROOT / "configs" / "explain_plan_brain.yaml", attribution_maps=[{"method": "grad_cam", "values": grad_cam, "target": predicted, "source_refs": ["reused measured normal-case Grad-CAM for controlled integrity route"]}], critical_fault=fault)
        target = output_root / control_id; target.mkdir(exist_ok=False); result.export_json(target / "result.json", detail="standard"); (target / "audit.json").write_text(json.dumps(result.audit(), indent=2) + "\n"); (target / "reader_ru.txt").write_text(result.full_report(level="reader")); result.visualize(view="provenance", selector="action", output=target / "provenance_action.png")
        system = result.system; summary[control_id] = {"rho": system.risk.rho, "candidate_action": system.risk.candidate_action, "critical_override": system.risk.critical_override, "final_action": system.risk.action, "diagnostics": list(system.diagnostics)}
    (output_root / "controls_summary.json").write_text(json.dumps(summary, indent=2) + "\n"); print(output_root)


if __name__ == "__main__":
    main()
