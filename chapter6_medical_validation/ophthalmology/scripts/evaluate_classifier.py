from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chapter6_medical_validation.ophthalmology.src.calibration import fit_temperature, probabilities_from_logits
from chapter6_medical_validation.ophthalmology.src.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute registered five-class metrics from saved predictions")
    parser.add_argument("predictions", type=Path, help="JSON with y_true and probabilities")
    parser.add_argument("output", type=Path)
    parser.add_argument("--fit-temperature", action="store_true", help="fit temperature on this declared validation artifact")
    args = parser.parse_args()
    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if rows is not None:
        truth = np.asarray([row["label"] for row in rows])
        probabilities = np.asarray([row["probabilities"] for row in rows])
        logits = np.asarray([row["logits"] for row in rows])
    else:
        truth = np.asarray(payload["y_true"])
        probabilities = np.asarray(payload["probabilities"])
        logits = np.asarray(payload["logits"]) if "logits" in payload else None
    metrics = {"uncalibrated": classification_metrics(truth, probabilities)}
    if args.fit_temperature:
        if logits is None:
            raise ValueError("temperature scaling requires saved logits")
        calibration = fit_temperature(logits, truth)
        metrics["temperature_scaling"] = calibration
        metrics["calibrated"] = classification_metrics(truth, probabilities_from_logits(logits, float(calibration["temperature"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
