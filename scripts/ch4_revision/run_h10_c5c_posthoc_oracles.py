#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fuzzyxai.experiments.h10_c5c_posthoc import (
    run_posthoc_oracle_decomposition,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--development-status",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_posthoc_oracle_decomposition(
        args.manifest.resolve(),
        args.development_status.resolve(),
        args.baseline_results.resolve(),
        args.root.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
