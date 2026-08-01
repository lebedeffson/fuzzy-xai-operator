#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fuzzyxai.experiments.h10_c5b import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", default=Path("."), type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest.resolve(), args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
