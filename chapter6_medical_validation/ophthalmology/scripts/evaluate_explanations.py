from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chapter6_medical_validation.ophthalmology.src.metrics import attribution_spatial_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved attribution against an IDRiD lesion mask")
    parser.add_argument("attribution", type=Path, help=".npy attribution tensor/map")
    parser.add_argument("lesion_mask", type=Path, help=".npy boolean lesion union")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = attribution_spatial_metrics(np.load(args.attribution), np.load(args.lesion_mask))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
