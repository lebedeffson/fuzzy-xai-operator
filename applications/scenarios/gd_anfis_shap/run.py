#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from applications.run_framework_scenario import run_framework_scenario

SCENARIO_ID = "gd_anfis_shap"

def main() -> None:
    run_framework_scenario(SCENARIO_ID, ROOT)

if __name__ == "__main__":
    main()
