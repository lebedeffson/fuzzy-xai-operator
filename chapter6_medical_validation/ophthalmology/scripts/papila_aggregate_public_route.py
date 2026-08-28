"""Create lightweight public FuzzyXAI route summaries for frozen PAPILA cohorts.

Selected cases retain their full LIME/Grad-CAM/IG artifacts.  This cohort-wide
pass intentionally does not repeat those expensive native XAI calculations:
it invokes the same public system route for every image and serializes only
system fields already computed in ``ModelExplanationResult``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fuzzyxai.adapters.contracts_v2 import LocalModelEvidence

from chapter6_medical_validation.ophthalmology.src.artifact_io import sha256_file, sha256_json
from chapter6_medical_validation.ophthalmology.src.datasets import load_yaml
from chapter6_medical_validation.ophthalmology.src.models import build_classifier
from chapter6_medical_validation.ophthalmology.src.papila import expert1_disc_roi
from chapter6_medical_validation.shared.fuzzyxai_adapter import SystemRunFacts, explain_public
from chapter6_medical_validation.shared.torch_adapter import TemperatureScaledTorchAdapter

ROOT = Path(__file__).resolve().parents[1]


class SystemRouteOnlyAdapter(TemperatureScaledTorchAdapter):  # type: ignore[misc]
    """Avoid recalculating native image maps in the explicitly lightweight pass."""

    def extract_local_evidence(self, inputs: Any, prediction: Any, context: Any) -> LocalModelEvidence:
        return LocalModelEvidence(
            channels={"contribution_method": "not_run_lightweight_system_route"},
            missing_channels=("native_attribution_not_repeated_for_aggregate_route",),
            limitations=("Full LIME/Grad-CAM artifacts are retained only for selected cases.",),
        )


def _labels(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["sample_id"]: row for row in csv.DictReader(stream)}


def _summary(result: Any, sample_id: str, cohort: str) -> dict[str, Any]:
    system = result.system
    assert system is not None
    risk = system.risk
    return {
        "sample_id": sample_id, "cohort": cohort,
        "prediction": int(np.asarray(result.prediction.predictions).reshape(-1)[0]),
        "system_Gamma": system.alignment["gamma"], "U_model": system.uncertainty.u_model,
        "U_trace": system.uncertainty.u_trace, "u_M": system.uncertainty.u_m,
        "I_pre": system.i_pre, "rho": risk.rho, "risk_status": risk.status,
        "candidate_action": risk.candidate_action, "critical_override": risk.critical_override,
        "final_action": risk.action,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_yaml(ROOT / "configs" / "preprocessing_papila.yaml")
    raw_root = next((args.data_root / "eyes" / "papila" / "raw").glob("PapilaDB-PAPILA-*"))
    contours = raw_root / "ExpertsSegmentations" / "Contours"
    labels = _labels(args.data_root / "eyes" / "papila" / "verified" / "papila_eye_labels.csv")
    test_rows = json.loads((args.run / "test_predictions.json").read_text(encoding="utf-8"))["rows"]
    suspect_rows = json.loads((args.data_root / "eyes" / "papila" / "suspect_predictions.json").read_text(encoding="utf-8"))["rows"]
    checkpoint_path = args.run / "best_model.pt"
    run = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    model = build_classifier("resnet50", num_classes=2, pretrained=True).cuda().eval()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cuda", weights_only=False)["state_dict"])

    def transform(value: object) -> torch.Tensor:
        image = np.asarray(value, dtype=np.float32).reshape(224, 224, 3)
        mean, std = np.asarray(cfg["mean"], dtype=np.float32), np.asarray(cfg["std"], dtype=np.float32)
        return torch.from_numpy(((image / 255.0 - mean) / std).transpose(2, 0, 1).astype(np.float32))[None].cuda()

    adapter = SystemRouteOnlyAdapter(model, temperature=1.0, task="classification", ig_steps=64, input_transform=transform)
    result_rows: list[dict[str, Any]] = []
    for cohort, source_rows in (("binary_outer_fold_5_test", test_rows), ("suspect_auxiliary", suspect_rows)):
        for source in source_rows:
            sample_id = str(source["sample_id"]); row = labels[sample_id]
            roi = expert1_disc_roi(args.data_root / row["image_path"], contours / f"{sample_id}_disc_exp1.txt", margin_fraction=float(cfg["roi_margin_fraction"]))
            display = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
            numeric = display.astype(np.float32).reshape(1, -1)
            prediction = adapter.predict(numeric); probabilities = tuple(float(value) for value in prediction.probabilities[0]); predicted = int(np.argmax(probabilities))
            facts = SystemRunFacts(
                "papila_figshare_14798004_v2", cohort, sample_id, row["image_sha256"],
                {"roi_source": cfg["roi_source"], "roi_extraction_version": cfg["roi_extraction_version"], "preprocessing_config_sha256": sha256_json(cfg)},
                "resnet50", sha256_file(checkpoint_path), run["run_id"], probabilities, predicted, 1, (row["image_sha256"],),
            )
            result = explain_public(model, adapter, numeric, display, facts, plan_path=ROOT / "configs" / "explain_plan_papila.yaml", attribution_maps=[])
            result_rows.append(_summary(result, sample_id, cohort))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": "1.0", "scope": "public lightweight route; native XAI not repeated", "rows": result_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
