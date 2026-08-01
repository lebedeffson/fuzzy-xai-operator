#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from fuzzyxai.robustness import run_cut_robustness

if __name__ == "__main__":
    print(json.dumps(run_cut_robustness(Path.cwd()), indent=2))
