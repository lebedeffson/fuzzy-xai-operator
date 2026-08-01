#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fuzzyxai.experiments.h10_c5c_data import prepare_bugsinpy_development


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bugsinpy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    result = prepare_bugsinpy_development(
        args.bugsinpy_root.resolve(),
        args.output.resolve(),
        args.cache.resolve(),
        args.root.resolve(),
        allow_network=args.allow_network,
    )
    print(
        json.dumps(
            {
                "status": "MATERIALIZED_AWAITING_RUNTIME_COLLECTION",
                "incident_count": result.incident_count,
                "repository_count": result.repository_count,
                "manifest_path": str(result.manifest_path),
                "command_registry_path": str(result.command_registry_path),
                "source_registry_path": str(result.source_registry_path),
                "selection_report_path": str(result.selection_report_path),
                "scientific_result": "NOT_EVALUATED",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
