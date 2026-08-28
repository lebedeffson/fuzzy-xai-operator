from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chapter6_medical_validation.ophthalmology.src.artifact_io import sha256_file
from chapter6_medical_validation.ophthalmology.src.datasets import configured_data_root, load_yaml
from chapter6_medical_validation.ophthalmology.src.evidence_adapter import MedicalRunFacts, export_public_result, run_public_explanation
from chapter6_medical_validation.ophthalmology.src.image_quality import technical_image_quality
from chapter6_medical_validation.ophthalmology.src.models import build_classifier
from chapter6_medical_validation.ophthalmology.src.preprocessing import preprocess_image

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import torch
    from fuzzyxai.adapters.optional_v2 import TorchAdapter

    parser = argparse.ArgumentParser(description="Run one CH6 case through the canonical public FuzzyXAI API")
    parser.add_argument("case", type=Path)
    parser.add_argument("--data-root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ig-steps", type=int, default=512)
    args = parser.parse_args()
    case = json.loads(args.case.read_text(encoding="utf-8"))
    data_root = configured_data_root(args.data_root)
    preprocess_cfg = load_yaml(ROOT / "configs" / "preprocessing_eye.yaml")
    architecture = str(case["architecture"])
    model = build_classifier(architecture, num_classes=5, pretrained=False)
    checkpoint = Path(case["checkpoint"])
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_payload["state_dict"])
    processed = preprocess_image(data_root / case["image_path"], preprocess_cfg, split=str(case["split"]))
    channels, height, width = processed.normalized_chw.shape
    flat_input = processed.normalized_chw.reshape(1, -1)
    adapter = TorchAdapter(
        model,
        task="classification",
        ig_steps=args.ig_steps,
        input_transform=lambda values: torch.as_tensor(values, dtype=torch.float32).reshape(-1, channels, height, width),
    )
    prediction = adapter.predict(flat_input)
    probability_values = tuple(float(value) for value in prediction.probabilities[0])
    if len(probability_values) != 5:
        raise ValueError("registered ophthalmology model must return five class probabilities")
    probabilities = probability_values
    quality = technical_image_quality(processed.rgb)
    quality["quality_score"] = 1.0
    facts = MedicalRunFacts(dataset_id=str(case["dataset_id"]), split=str(case["split"]), sample_id=str(case["sample_id"]), raw_image_sha256=processed.trace["source_sha256"], preprocessing_trace=processed.trace, architecture=architecture, checkpoint_sha256=sha256_file(checkpoint), model_run_id=str(checkpoint_payload["metadata"]["run_id"]), full_probabilities=probabilities, predicted_grade=int(np.argmax(probabilities)), calibration_status=str(case.get("calibration_status", "not_calibrated")), technical_quality=quality)
    cam_dir = Path(case["grad_cam_dir"])
    cam_metadata = json.loads((cam_dir / "grad_cam.json").read_text(encoding="utf-8"))
    result = run_public_explanation(model, adapter, flat_input, processed.rgb, facts, plan_path=ROOT / "configs" / "explain_plan_eye.yaml", grad_cam_map=np.load(cam_dir / "grad_cam_raw.npy"), grad_cam_metadata=cam_metadata, lesion_masks=None, critical_fault=case.get("critical_fault"))
    export_public_result(result, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
