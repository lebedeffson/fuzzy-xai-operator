#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from fuzzyxai.experiments.h9_e2e_v2 import run

if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2))
