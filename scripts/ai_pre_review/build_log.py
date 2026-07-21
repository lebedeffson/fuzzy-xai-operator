#!/usr/bin/env python3
"""Build the blind master log and deterministic review batches."""

from __future__ import annotations

import json
from pathlib import Path

from fuzzyxai.ai_pre_review import build_study_inputs


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    result = build_study_inputs(ROOT)
    print(json.dumps({"status": "PASS", "cases": result["cases"], "variants": result["variants"], "batches": len(result["batches"])}, sort_keys=True))
