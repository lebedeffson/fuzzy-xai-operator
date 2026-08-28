from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from chapter6_medical_validation.brain_allen.src.model import build_inception_binary
from chapter6_medical_validation.brain_allen.src.preprocessing import preprocess_patch
from chapter6_medical_validation.ophthalmology.src.metrics import attribution_spatial_metrics
from chapter6_medical_validation.ophthalmology.src.native_xai import grad_cam
from chapter6_medical_validation.shared.fuzzyxai_adapter import SystemRunFacts, explain_public
from chapter6_medical_validation.shared.torch_adapter import TemperatureScaledTorchAdapter

ROOT = Path(__file__).resolve().parents[1]


def canonical_run(output_root: str = "outputs") -> tuple[Path, dict]:
    candidates = []
    for path in (ROOT / output_root / "runs").glob("*/run.json"):
        payload = json.loads(path.read_text()); candidates.append((min(row["validation_loss"] for row in payload["history"]), -max(row["validation_macro_f1"] for row in payload["history"]), path, payload))
    if not candidates: raise FileNotFoundError("no brain training runs")
    _, _, path, payload = min(candidates); return path.parent, payload


def main() -> None:
    from chapter6_medical_validation.ophthalmology.src.artifact_io import sha256_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-name", default="prepared")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--protocol-id", default="brain_v1_pilot")
    parser.add_argument("--selection-path", type=Path)
    parser.add_argument("--object-prefix", default="allen")
    args = parser.parse_args()
    data_root = os.environ.get("FUZZYXAI_CH6_DATA_ROOT")
    if not data_root: raise FileNotFoundError("FUZZYXAI_CH6_DATA_ROOT is not set")
    prepared = Path(data_root) / "brain" / "allen_ccf_25um" / args.prepared_name; patches = np.load(prepared / "patches.npy", mmap_mode="r"); masks = np.load(prepared / "hpf_masks.npy", mmap_mode="r"); metadata = json.loads((prepared / "patches.json").read_text())
    run_dir, run = canonical_run(args.output_root); prediction_data = np.load(run_dir / "test_predictions.npz"); checkpoint_path = run_dir / "best.pt"; checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    # The training builder uses pretrained=True, which also fixes torchvision's
    # non-state-dict input-transform behavior.  Public replay must reproduce
    # that constructor contract before loading the checkpoint.
    model = build_inception_binary(pretrained=True).cuda().eval(); model.load_state_dict(checkpoint["state_dict"]); scale = float(run["preprocessing"]["scale"]); temperature = float(run["calibration"]["temperature"])
    output_root = ROOT / args.output_root / "cases"; output_root.mkdir(parents=True, exist_ok=True); patch_shape = tuple(int(value) for value in patches.shape[1:]); summaries = []
    selected_indices = None
    if args.selection_path is not None:
        selected = json.loads(args.selection_path.read_text(encoding="utf-8"))["cases"]
        selected_indices = {int(value["prepared_index"]) for value in selected.values() if isinstance(value, dict) and "prepared_index" in value}
    for prepared_index in prediction_data["prepared_indices"]:
        index = int(prepared_index); item = metadata[index]; tensor, display, trace = preprocess_patch(np.asarray(patches[index]), scale=scale, split="test", seed=int(run["seed"]), object_index=index); tensor = tensor[None].cuda(); numeric_input = np.asarray(patches[index], dtype=np.float32).reshape(1, -1)
        if selected_indices is not None and index not in selected_indices:
            continue
        def input_transform(value: object, *, scale: float = scale, seed: int = int(run["seed"]), index: int = index) -> torch.Tensor:
            raw_patch = np.asarray(value, dtype=np.float32).reshape(patch_shape)
            transformed, _, _ = preprocess_patch(raw_patch, scale=scale, split="test", seed=seed, object_index=index)
            return transformed[None].cuda()
        adapter = TemperatureScaledTorchAdapter(model, temperature=temperature, task="classification", ig_steps=64, input_transform=input_transform); prediction = adapter.predict(numeric_input); probabilities = tuple(float(value) for value in prediction.probabilities[0]); predicted = int(np.argmax(probabilities)); object_id = f"{args.object_prefix}-section-{item['section']}-r{item['row']}-c{item['col']}"
        cam = grad_cam(model, tensor, model.Mixed_7c, target_layer_id="Mixed_7c", sample_id=object_id, checkpoint_sha256=sha256_file(checkpoint_path), target_class=predicted)
        facts = SystemRunFacts(f"allen-ccf-2017-25um:{args.protocol_id}", "test", object_id, item["hash"], trace, "inception_v3", sha256_file(checkpoint_path), run["run_id"], probabilities, predicted, 1, (item["hash"],))
        result = explain_public(model, adapter, numeric_input, display, facts, plan_path=ROOT / "configs" / "explain_plan_brain.yaml", attribution_maps=[{"method": "grad_cam", "values": cam.raw_map, "target": predicted, "source_refs": ["Mixed_7c"]}], technical_evidence={"patch_variance": float(np.var(patches[index])), "gray_fraction": float(item["gray_fraction"]), "hpf_fraction": float(item["hpf_fraction"])})
        case_dir = output_root / object_id; case_dir.mkdir(exist_ok=False); result.export_json(case_dir / "result.json", detail="standard"); (case_dir / "audit.json").write_text(json.dumps(result.audit(), indent=2) + "\n"); (case_dir / "reader_ru.txt").write_text(result.full_report(level="reader")); (case_dir / "audit_ru.txt").write_text(result.full_report(level="audit")); (case_dir / "provenance_action.json").write_text(json.dumps(result.inspect("action").to_dict(), indent=2) + "\n"); result.visualize(view="provenance", selector="action", output=case_dir / "provenance_action.png"); np.save(case_dir / "grad_cam_raw.npy", cam.raw_map)
        raw_payload = result.to_dict(detail="audit", include_raw=True); maps = raw_payload["layers"]["attribution_maps"]; ig_entry = next(value for value in maps if value["method"] == "integrated_gradients"); ig = np.asarray(ig_entry["attribution_array"], dtype=np.float32).reshape(3, 299, 299); np.save(case_dir / "integrated_gradients_signed.npy", ig); positive_ig = np.maximum(ig, 0).sum(axis=0); mask = cv2.resize(np.asarray(masks[index], dtype=np.uint8), (299, 299), interpolation=cv2.INTER_NEAREST).astype(bool); diagnostics = {"grad_cam": attribution_spatial_metrics(cam.raw_map, mask), "positive_ig": attribution_spatial_metrics(positive_ig, mask), "semantics": "anatomical_spatial_diagnostic_not_Gamma_or_causality"} if item["label"] == 1 else {"status": "not_applicable", "reason": "HPF overlap is defined only for positive HPF reference patches"}; (case_dir / "spatial_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        system = result.system
        confidence = max(probabilities)
        patch_std = float(np.std(patches[index]))
        technical_quality = min(float(item["gray_fraction"]), min(1.0, patch_std / (0.10 * scale)))
        summaries.append({"object_id": object_id, "prepared_index": index, "label": item["label"], "prediction": predicted, "probabilities": probabilities, "confidence": confidence, "top1_top2_margin": float(np.diff(np.sort(probabilities)[-2:])[0]), "correct": predicted == item["label"], "technical_quality_score": technical_quality, "technical_quality_semantics": "min(gray_fraction, patch_std/(0.10*training_scale))", "gamma": system.alignment["gamma"], "u_M": system.uncertainty.u_m, "Delta_status": system.reduction_status, "i_pre": system.i_pre, "rho": system.risk.rho, "candidate_action": system.risk.candidate_action, "critical": system.risk.critical_override, "action": system.risk.action})
    (output_root / "case_summaries.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(output_root)


if __name__ == "__main__": main()
