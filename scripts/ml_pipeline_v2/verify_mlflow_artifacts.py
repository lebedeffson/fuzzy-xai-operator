#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote, urlparse

from fuzzyxai.ml_vertical.tracking import PIPELINE_ARTIFACTS

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/ml_pipeline_v2"


def main() -> int:
    runs = json.loads((RESULTS / "MLFLOW_RUNS.json").read_text(encoding="utf-8"))
    per_run = []
    for run in runs:
        parsed = urlparse(run["artifact_uri"])
        artifact_root = Path(unquote(parsed.path)) / "pipeline"
        missing = sorted(name for name in PIPELINE_ARTIFACTS if not (artifact_root / name).is_file())
        per_run.append({"run_id": run["run_id"], "artifact_uri": run["artifact_uri"], "missing": missing})
    complete = sum(not row["missing"] for row in per_run)
    payload = {
        "status": "PASS" if len(runs) == 18 and complete == 18 else "FAIL",
        "run_count": len(runs),
        "complete_runs": complete,
        "required_artifacts_per_run": len(PIPELINE_ARTIFACTS),
        "runs": per_run,
    }
    (RESULTS / "MLFLOW_ARTIFACT_AUDIT.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "runs"}, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
