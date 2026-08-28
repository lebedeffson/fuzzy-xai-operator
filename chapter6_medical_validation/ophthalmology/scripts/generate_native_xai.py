from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chapter6_medical_validation.ophthalmology.src.artifact_io import sha256_file
from chapter6_medical_validation.ophthalmology.src.datasets import configured_data_root, load_yaml
from chapter6_medical_validation.ophthalmology.src.models import build_classifier, resolve_module
from chapter6_medical_validation.ophthalmology.src.native_xai import grad_cam
from chapter6_medical_validation.ophthalmology.src.preprocessing import preprocess_image

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description="Generate native Grad-CAM for a frozen selected case")
    parser.add_argument("case", type=Path, help="JSON with sample_id, image_path, architecture, checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case = json.loads(args.case.read_text(encoding="utf-8"))
    data_root = configured_data_root(args.data_root)
    architecture = str(case["architecture"])
    model_cfg = load_yaml(ROOT / "configs" / f"model_{architecture.replace('_', '')}.yaml")
    preprocess_cfg = load_yaml(ROOT / "configs" / "preprocessing_eye.yaml")
    model = build_classifier(architecture, num_classes=5, pretrained=False)
    checkpoint = Path(case["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    processed = preprocess_image(data_root / case["image_path"], preprocess_cfg, split=str(case["split"]))
    tensor = torch.from_numpy(processed.normalized_chw[None, ...])
    result = grad_cam(model, tensor, resolve_module(model, model_cfg["grad_cam_target_layer"]), target_layer_id=model_cfg["grad_cam_target_layer"], sample_id=str(case["sample_id"]), checkpoint_sha256=sha256_file(checkpoint), target_class=case.get("target_class"))
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / "grad_cam_raw.npy", result.raw_map)
    np.save(args.output / "grad_cam_normalized.npy", result.normalized_map)
    (args.output / "grad_cam.json").write_text(json.dumps(result.metadata(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "preprocessing_trace.json").write_text(json.dumps(processed.trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
