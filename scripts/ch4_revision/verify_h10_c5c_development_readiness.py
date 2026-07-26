#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fuzzyxai.experiments.h10_c5c_readiness import (
    verify_h10_c5c_development_readiness,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--command-registry", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = verify_h10_c5c_development_readiness(
        args.manifest.resolve(),
        args.command_registry.resolve(),
        args.source_registry.resolve(),
        args.runtime_report.resolve(),
        args.output.resolve(),
        args.root.resolve(),
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "incident_count": result.incident_count,
                "repository_count": result.repository_count,
                "report_path": str(result.report_path),
                "scientific_result": "NOT_EVALUATED",
            },
            indent=2,
            sort_keys=True,
        )
    )
    if result.status != "H10_C5C_DEVELOPMENT_READINESS_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
