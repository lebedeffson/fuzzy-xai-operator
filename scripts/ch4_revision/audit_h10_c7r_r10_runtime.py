#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fuzzyxai.experiments.h10_c7r import read_jsonl
from fuzzyxai.experiments.h10_c7r_r10 import (
    audit_runtime_file,
    summarize_runtime_readiness,
)


def _resolve(base: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed readiness audit for R10 causal runtime events"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for incident in read_jsonl(args.manifest):
        runtime_path = _resolve(
            args.manifest.parent,
            incident["runtime_events_path"],
        )
        readiness = audit_runtime_file(runtime_path)
        rows.append(
            {
                "incident_id": str(incident["incident_id"]),
                "repository": str(incident["repository"]),
                "runtime_events_path": str(runtime_path),
                **readiness.to_mapping(),
            }
        )

    summary = summarize_runtime_readiness(rows)
    payload = {"summary": summary, "incidents": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["all_incidents_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
