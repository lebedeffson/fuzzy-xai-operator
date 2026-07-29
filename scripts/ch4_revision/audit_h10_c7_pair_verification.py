#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fuzzyxai.experiments.h10_c7_pair_verification import (
    run_pair_verification_audits,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit H10-C7 R5V pair targets and observable evidence."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_pair_verification_audits(
        bundle=args.bundle.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
