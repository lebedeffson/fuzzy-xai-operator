#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fuzzyxai.experiments.h10_c5c_runtime import collect_h10_c5c_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--command-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--allow-setup", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--interpreter-map", type=Path)
    args = parser.parse_args()
    interpreter_map = None
    if args.interpreter_map is not None:
        interpreter_map = json.loads(args.interpreter_map.read_text(encoding="utf-8"))
        if not isinstance(interpreter_map, dict):
            raise ValueError("interpreter map must be a JSON object")
        interpreter_map = {str(key): str(value) for key, value in interpreter_map.items()}
    result = collect_h10_c5c_runtime(
        args.manifest.resolve(),
        args.command_registry.resolve(),
        args.output.resolve(),
        timeout_seconds=args.timeout_seconds,
        allow_setup=args.allow_setup,
        interpreter_map=interpreter_map,
        max_workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": (
                    "DEVELOPMENT_RUNTIME_COMPLETE"
                    if result.complete_incidents == result.total_incidents
                    else "DEVELOPMENT_RUNTIME_INCOMPLETE"
                ),
                "complete_incidents": result.complete_incidents,
                "total_incidents": result.total_incidents,
                "enriched_manifest_path": str(result.enriched_manifest_path),
                "report_path": str(result.report_path),
                "scientific_result": "NOT_EVALUATED",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
