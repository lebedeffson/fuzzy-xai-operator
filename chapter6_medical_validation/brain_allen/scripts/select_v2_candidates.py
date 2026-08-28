"""Choose v2 public-XAI candidates from frozen canonical test predictions.

Selection uses only labels, calibrated probabilities and registered technical
metadata.  It deliberately performs no native-XAI or system calculation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from chapter6_medical_validation.brain_allen.scripts.run_cases import canonical_run
from chapter6_medical_validation.shared.calibration import softmax

ROOT = Path(__file__).resolve().parents[1]


def _row(index: int, label: int, probability: np.ndarray, metadata: list[dict[str, object]]) -> dict[str, object]:
    prediction = int(np.argmax(probability))
    item = metadata[index]
    return {
        "prepared_index": index,
        "object_id": f"allen-v2-section-{item['section']}-r{item['row']}-c{item['col']}",
        "label": label,
        "prediction": prediction,
        "confidence": float(probability[prediction]),
        "margin": float(np.diff(np.sort(probability)[-2:])[0]),
        "correct": prediction == label,
        "gray_fraction": float(item["gray_fraction"]),
        "hpf_fraction": float(item["hpf_fraction"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-name", default="prepared_v2_confirmatory")
    parser.add_argument("--output-root", default="outputs_v2_confirmatory")
    args = parser.parse_args()
    data_root = os.environ.get("FUZZYXAI_CH6_DATA_ROOT")
    if not data_root:
        raise FileNotFoundError("FUZZYXAI_CH6_DATA_ROOT is not set")
    prepared = Path(data_root) / "brain" / "allen_ccf_25um" / args.prepared_name
    metadata = json.loads((prepared / "patches.json").read_text(encoding="utf-8"))
    run_dir, run = canonical_run(args.output_root)
    prediction_data = np.load(run_dir / "test_predictions.npz")
    probabilities = softmax(prediction_data["logits"], float(run["calibration"]["temperature"]))
    rows = [_row(int(index), int(prediction_data["labels"][position]), probabilities[position], metadata) for position, index in enumerate(prediction_data["prepared_indices"])]
    correct_hpf = [row for row in rows if row["correct"] and row["label"] == 1]
    correct_other = [row for row in rows if row["correct"] and row["label"] == 0]
    errors = [row for row in rows if not row["correct"]]
    selected = {
        "BRAIN_V2_A": max(correct_hpf, key=lambda row: float(row["confidence"])),
        "BRAIN_V2_B": max(correct_other, key=lambda row: float(row["confidence"])),
        "BRAIN_V2_C": min(rows, key=lambda row: float(row["margin"])),
        "BRAIN_V2_D": max(errors, key=lambda row: float(row["confidence"])) if errors else {"status": "not_available", "reason": "no canonical v2 test errors"},
        "BRAIN_V2_E": min(rows, key=lambda row: float(row["gray_fraction"])),
    }
    output = ROOT / args.output_root / "cases" / "selected_candidates.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"protocol_id": "brain_v2_confirmatory", "selection_policy": "frozen canonical calibrated test predictions and factual metadata only", "canonical_run_id": run["run_id"], "cases": selected}, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
