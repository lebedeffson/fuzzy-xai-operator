#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from fuzzyxai.experiments.h10_c5_pilot import (
    run_pilot_selection,
)


def main() -> None:
    print(
        json.dumps(
            run_pilot_selection(Path.cwd()),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
