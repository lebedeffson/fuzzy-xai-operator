from __future__ import annotations

import argparse
import json
from pathlib import Path

from fuzzyxai.experiments.h10_c7_confirmation import run_r5c_confirmation
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedNaturalDiagnosisEngine,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run H10-C7-R5C LORO confirmation on the open replay"
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    result = run_r5c_confirmation(
        bundle=args.bundle.resolve(),
        output=args.output.resolve(),
        reports=args.reports.resolve(),
        engine=GuidedNaturalDiagnosisEngine(structural_only=True),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
