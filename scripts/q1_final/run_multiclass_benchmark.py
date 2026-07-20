#!/usr/bin/env python3
"""CLI for one native multiclass modality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fuzzyxai.q1_final.multiclass import run_native_multiclass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=("tabular", "image", "text", "timeseries"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    payload = run_native_multiclass(args.modality, args.output, args.cache)
    evaluation_path = args.output.parent / f"{args.modality}_evaluation_object_ids.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "dataset_id": payload["dataset"]["dataset_id"],
                "modality": args.modality,
                "object_ids": payload["evaluation_object_ids"],
                "frozen_before_explainer_comparison": True,
                "sampling": payload["evaluation_sampling"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"PASS: q1_final_multiclass modality={args.modality} "
        f"objects={payload['dataset']['n_objects']} runs={len(payload['models'])}"
    )


if __name__ == "__main__":
    main()
