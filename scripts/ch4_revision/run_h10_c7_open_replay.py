#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fuzzyxai.experiments.h10_c7_replay import run_open_replay_tournament
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedNaturalDiagnosisEngine,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_open_replay_tournament(
        bundle=args.bundle,
        output=args.output,
        engine=GuidedNaturalDiagnosisEngine(structural_only=True),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
