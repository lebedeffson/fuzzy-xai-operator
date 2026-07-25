#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from fuzzyxai.multimodal import run_validation

if __name__ == "__main__":
    print(json.dumps(run_validation(Path.cwd()), indent=2))
