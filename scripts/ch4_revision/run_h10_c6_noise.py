#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fuzzyxai.experiments.h10_c6_noise import prepare_noise_protocol, run_noise_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = prepare_noise_protocol(args.root.resolve()) if args.operation == "prepare" else run_noise_experiment(args.root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
