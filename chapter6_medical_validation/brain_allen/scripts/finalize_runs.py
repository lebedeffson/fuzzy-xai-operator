from __future__ import annotations

import argparse
import json
from pathlib import Path

from chapter6_medical_validation.shared.calibration import softmax
from chapter6_medical_validation.shared.metrics_common import binary_metrics

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()
    for run_json in sorted((ROOT / args.output_root / "runs").glob("*/run.json")):
        payload = json.loads(run_json.read_text()); temperature = float(payload["calibration"]["temperature"])
        payload.setdefault("preprocessing", {})["normalization"] = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "semantics": "frozen symmetric transform used during training"}
        for split in ("validation", "test"):
            import numpy as np

            data = np.load(run_json.parent / f"{split}_predictions.npz")
            payload[f"{split}_metrics_uncalibrated"] = binary_metrics(data["labels"], softmax(data["logits"])[:, 1])
            payload[f"{split}_metrics_calibrated"] = binary_metrics(data["labels"], softmax(data["logits"], temperature)[:, 1])
        payload.pop("validation_metrics", None); payload.pop("test_metrics", None)
        run_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__": main()
