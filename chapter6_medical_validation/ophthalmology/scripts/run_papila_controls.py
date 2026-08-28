"""Run registered PAPILA integrity controls through the public API.

The frozen EYE_A image, model checkpoint, preprocessing and native maps are
reused.  Each altered fact is declared in ``critical_fault`` so the result is
a controlled route-integrity experiment, not a claim about the patient.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from chapter6_medical_validation.ophthalmology.src.artifact_io import sha256_file, sha256_json
from chapter6_medical_validation.ophthalmology.src.datasets import load_yaml
from chapter6_medical_validation.ophthalmology.src.models import build_classifier
from chapter6_medical_validation.ophthalmology.src.papila import expert1_disc_roi
from chapter6_medical_validation.shared.fuzzyxai_adapter import SystemRunFacts, explain_public
from chapter6_medical_validation.shared.torch_adapter import TemperatureScaledTorchAdapter

ROOT = Path(__file__).resolve().parents[1]


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["sample_id"]: row for row in csv.DictReader(stream)}


def _summary(result: Any, *, description: str) -> dict[str, Any]:
    system = result.system
    assert system is not None
    risk = system.risk
    audit = system.audit_dict()
    return {
        "description": description,
        "gamma": system.alignment.get("gamma"),
        "gamma_status": system.alignment.get("status", "measured"),
        "u_trace": system.uncertainty.u_trace,
        "u_m": system.uncertainty.u_m,
        "i_pre": system.i_pre,
        "rho": risk.rho,
        "candidate_action": risk.candidate_action,
        "critical_override": risk.critical_override,
        "final_action": risk.action,
        "missing_required": list(risk.__dict__.get("missing_required_components", ())),
        "optional_missing": list(getattr(result.view_model, "optional_missing_channels", ())),
        "diagnostics": list(system.diagnostics),
        "audit_reason": [item.get("reason", item.get("code")) for item in system.diagnostics],
        "prediction": result.prediction.predictions,
        "system_audit_ref": audit["system_source"]["source_refs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PAPILA controlled public-route integrity faults")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(ROOT / "configs" / "preprocessing_papila.yaml")
    raw_root = next((args.data_root / "eyes" / "papila" / "raw").glob("PapilaDB-PAPILA-*"))
    contours = raw_root / "ExpertsSegmentations" / "Contours"
    labels = _read_rows(args.data_root / "eyes" / "papila" / "verified" / "papila_eye_labels.csv")
    sample_id = "RET038OS"  # frozen EYE_A
    row = labels[sample_id]
    checkpoint_path = args.run / "best_model.pt"
    run = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    model = build_classifier("resnet50", num_classes=2, pretrained=True).cuda().eval()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cuda", weights_only=False)["state_dict"])
    raw_roi = expert1_disc_roi(
        args.data_root / row["image_path"], contours / f"{sample_id}_disc_exp1.txt",
        margin_fraction=float(cfg["roi_margin_fraction"]),
    )
    display = cv2.resize(raw_roi, (224, 224), interpolation=cv2.INTER_AREA)

    def transform(value: object) -> torch.Tensor:
        image = np.asarray(value, dtype=np.float32).reshape(224, 224, 3)
        mean, std = np.asarray(cfg["mean"], dtype=np.float32), np.asarray(cfg["std"], dtype=np.float32)
        return torch.from_numpy(((image / 255.0 - mean) / std).transpose(2, 0, 1).astype(np.float32))[None].cuda()

    adapter = TemperatureScaledTorchAdapter(model, temperature=1.0, task="classification", ig_steps=64, input_transform=transform)
    numeric = display.astype(np.float32).reshape(1, -1)
    prediction = adapter.predict(numeric)
    probabilities = tuple(float(value) for value in prediction.probabilities[0])
    target = int(np.argmax(probabilities))
    facts = SystemRunFacts(
        "papila_figshare_14798004_v2", "outer_fold_5_test", sample_id, row["image_sha256"],
        {"roi_source": cfg["roi_source"], "roi_extraction_version": cfg["roi_extraction_version"],
         "preprocessing_config_sha256": sha256_json(cfg)},
        "resnet50", sha256_file(checkpoint_path), run["run_id"], probabilities, target, 1,
        (row["image_sha256"],),
    )
    source = args.cases_root / sample_id
    lime = np.load(source / "lime_positive_map.npy")
    cam = np.load(source / "grad_cam_raw.npy")
    maps = [
        {"method": "lime_positive_support", "values": lime, "target": target, "source_refs": ["papila_lime_slic_v1"]},
        {"method": "grad_cam", "values": cam, "target": target, "source_refs": ["layer4.2.conv3"]},
    ]
    plan_path = ROOT / "configs" / "explain_plan_papila.yaml"
    fault_cases: list[tuple[str, str, SystemRunFacts, list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]] = [
        ("CONTROL_0", "normal frozen route", facts, maps, None, {"expert2_segmentation": "available_or_not_required"}),
        ("CONTROL_1", "Grad-CAM source omitted; frozen plan declares it required", facts, maps[:1], None, {"grad_cam": "omitted"}),
        ("CONTROL_2", "controlled missing required preprocessing provenance", facts, maps,
         {"critical": True, "missing_required_trace": True, "code": "controlled_missing_preprocessing_provenance", "source_refs": [sample_id, "preprocessing_config_sha256"]}, {}),
        ("CONTROL_3", "controlled checkpoint hash mismatch", replace(facts, checkpoint_sha256="0" * 64), maps,
         {"critical": True, "code": "controlled_checkpoint_hash_mismatch", "source_refs": [sample_id, facts.checkpoint_sha256]}, {}),
        ("CONTROL_4", "controlled LIME target mismatch", facts, [{**maps[0], "target": 1 - target}, maps[1]],
         {"critical": True, "code": "controlled_lime_target_mismatch", "source_refs": [sample_id, f"prediction:{target}", f"lime_target:{1-target}"]}, {}),
        ("CONTROL_5", "controlled image-to-patient linkage mismatch", facts, maps,
         {"critical": True, "code": "controlled_image_patient_linkage_mismatch", "source_refs": [sample_id, "registered_patient_linkage_mismatch"]}, {}),
        ("CONTROL_6", "registered non-critical native-XAI disagreement", facts, maps,
         {"critical": False, "code": "controlled_noncritical_native_xai_disagreement", "source_refs": [sample_id, "registered_lime_gradcam_diagnostic"]}, {"native_xai_disagreement": 1.0}),
        ("CONTROL_7", "Expert-2 segmentation omitted; declared optional by frozen plan", facts, maps, None, {"expert2_segmentation": "omitted_optional"}),
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for control_id, description, control_facts, control_maps, fault, technical in fault_cases:
        result = explain_public(
            model, adapter, numeric, display, control_facts, plan_path=plan_path,
            attribution_maps=control_maps, technical_evidence=technical, critical_fault=fault,
        )
        if int(np.asarray(result.prediction.predictions).reshape(-1)[0]) != target:
            raise AssertionError("public FuzzyXAI prediction changed under a route-only integrity control")
        destination = args.output / control_id
        destination.mkdir(exist_ok=False)
        result.export_json(destination / "result.json", detail="audit")
        (destination / "audit.json").write_text(json.dumps(result.audit(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (destination / "reader_ru.txt").write_text(result.full_report(level="reader"), encoding="utf-8")
        (destination / "inspect_action.json").write_text(json.dumps(result.inspect("action").to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result.visualize(view="provenance", selector="action", output=destination / "provenance_action.png")
        summary[control_id] = _summary(result, description=description)
    (args.output / "controls_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
