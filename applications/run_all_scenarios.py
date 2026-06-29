#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ["hybrid_xiris", "medical_ecg_signal", "gd_anfis_shap", "beacon_xai", "gis_integro"]

def main() -> None:
    for sid in SCENARIOS:
        script = ROOT / "applications" / "scenarios" / sid / "run.py"
        print(f"\n== {sid} ==")
        subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)

if __name__ == "__main__":
    main()
