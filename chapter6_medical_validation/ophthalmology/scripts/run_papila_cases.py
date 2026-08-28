"""Create PAPILA native-XAI and public FuzzyXAI artifacts from frozen fold 5."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from chapter6_medical_validation.ophthalmology.src.artifact_io import sha256_file, sha256_json
from chapter6_medical_validation.ophthalmology.src.datasets import configured_data_root, load_yaml
from chapter6_medical_validation.ophthalmology.src.lime_image import explain_lime
from chapter6_medical_validation.ophthalmology.src.models import build_classifier, resolve_module
from chapter6_medical_validation.ophthalmology.src.native_xai import grad_cam
from chapter6_medical_validation.ophthalmology.src.papila import expert1_disc_roi, papila_tensor
from chapter6_medical_validation.shared.fuzzyxai_adapter import SystemRunFacts, explain_public
from chapter6_medical_validation.shared.torch_adapter import TemperatureScaledTorchAdapter

ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream: return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run registered PAPILA LIME + Grad-CAM + public FuzzyXAI cases")
    parser.add_argument("--data-root")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); root = configured_data_root(args.data_root); cfg = load_yaml(ROOT / "configs" / "preprocessing_papila.yaml")
    checkpoint_path = args.run / "best_model.pt"; run = json.loads((args.run / "run.json").read_text(encoding="utf-8")); checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model = build_classifier("resnet50", num_classes=2, pretrained=True).cuda().eval(); model.load_state_dict(checkpoint["state_dict"])
    raw_root = next((root / "eyes" / "papila" / "raw").glob("PapilaDB-PAPILA-*")); contours = raw_root / "ExpertsSegmentations" / "Contours"; labels = {row["sample_id"]: row for row in _rows(root / "eyes" / "papila" / "verified" / "papila_eye_labels.csv")}
    output = args.output or root / "eyes" / "papila" / "cases"; output.mkdir(parents=True, exist_ok=True)
    for sample_id in args.sample_id:
        row = labels[sample_id]; image_path = root / row["image_path"]; contour = contours / f"{sample_id}_disc_exp1.txt"; raw_roi = expert1_disc_roi(image_path, contour, margin_fraction=float(cfg["roi_margin_fraction"])); display = cv2.resize(raw_roi, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = papila_tensor(image_path, contour, cfg, training=False, seed=None)[None]
        def transform(value: object) -> torch.Tensor:
            image = np.asarray(value, dtype=np.float32).reshape(224, 224, 3)
            mean, std = np.asarray(cfg["mean"], dtype=np.float32), np.asarray(cfg["std"], dtype=np.float32)
            return torch.from_numpy(((image / 255.0 - mean) / std).transpose(2, 0, 1).astype(np.float32))[None].cuda()
        def probabilities(images: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                values = np.asarray(images, dtype=np.float32); batch = np.stack([transform(item)[0].cpu().numpy() for item in values]); logits = model(torch.from_numpy(batch).cuda()); return torch.softmax(logits, dim=1).cpu().numpy()
        adapter = TemperatureScaledTorchAdapter(model, temperature=1.0, task="classification", ig_steps=64, input_transform=transform)
        numeric = display.astype(np.float32).reshape(1, -1); prediction = adapter.predict(numeric); prob = tuple(float(item) for item in prediction.probabilities[0]); target = int(np.argmax(prob)); checkpoint_sha = sha256_file(checkpoint_path)
        lime = explain_lime(display, probabilities, target_class=target)
        cam = grad_cam(model, torch.from_numpy(tensor).cuda(), resolve_module(model, "layer4.2.conv3"), target_layer_id="layer4.2.conv3", sample_id=sample_id, checkpoint_sha256=checkpoint_sha, target_class=target)
        facts = SystemRunFacts("papila_figshare_14798004_v2", "outer_fold_5_test", sample_id, row["image_sha256"], {"roi_source": cfg["roi_source"], "roi_extraction_version": cfg["roi_extraction_version"], "preprocessing_config_sha256": sha256_json(cfg)}, "resnet50", checkpoint_sha, run["run_id"], prob, target, 1, (row["image_sha256"],))
        result = explain_public(model, adapter, numeric, display, facts, plan_path=ROOT / "configs" / "explain_plan_papila.yaml", attribution_maps=[{"method": "lime_positive_support", "values": lime.positive_map, "target": target, "source_refs": ["papila_lime_slic_v1"]}, {"method": "grad_cam", "values": cam.raw_map, "target": target, "source_refs": [cam.target_layer]}], technical_evidence={"roi_source": cfg["roi_source"], "lime_local_fit_r2": lime.local_fit_r2, "lime_negative_superpixels": len(lime.negative_coefficients)})
        case_dir = output / sample_id; case_dir.mkdir(exist_ok=False); result.export_json(case_dir / "result.json", detail="audit"); (case_dir / "reader_ru.txt").write_text(result.full_report(level="reader"), encoding="utf-8"); (case_dir / "audit_ru.txt").write_text(result.full_report(level="audit"), encoding="utf-8"); (case_dir / "audit.json").write_text(json.dumps(result.audit(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); (case_dir / "provenance_action.json").write_text(json.dumps(result.inspect("action").to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); result.visualize(view="provenance", selector="action", output=case_dir / "provenance_action.png")
        np.save(case_dir / "lime_signed_map.npy", lime.signed_map); np.save(case_dir / "lime_positive_map.npy", lime.positive_map); np.save(case_dir / "grad_cam_raw.npy", cam.raw_map); np.save(case_dir / "lime_superpixels.npy", lime.superpixels)
        (case_dir / "lime.json").write_text(json.dumps({"target": target, "intercept": lime.intercept, "local_fit_r2": lime.local_fit_r2, "coefficients": lime.coefficients.tolist(), "negative_coefficients": lime.negative_coefficients, "perturbation_hash": lime.perturbation_hash, "parameters": {"segments": 50, "perturbations": 1000, "seed": 2026}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__": main()
