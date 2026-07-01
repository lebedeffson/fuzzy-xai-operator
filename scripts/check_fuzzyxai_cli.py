#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


PAYLOAD = {
    "scenario_id": "external_wine_classification",
    "source_type": "tabular",
    "model_name": "CLISmokeModel",
    "dataset_name": "cli_smoke",
    "predicted_class": 1,
    "class_probability": 0.68,
    "feature_values": {"x1": 0.4, "x2": 0.8, "x3": 0.2},
    "feature_importance": {"x1": 0.31, "x2": 0.26, "x3": 0.15},
    "quality_metrics": {"missing_rate": 0.0, "feature_range_violation": 0.0},
    "context_values": {"external_task": True, "task_type": "tabular_classification"},
}


def run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    print(result.stdout, end="")
    if result.returncode:
        print(result.stderr, end="", file=sys.stderr)
        raise SystemExit(result.returncode)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="fuzzyxai-cli-") as tmp:
        root = Path(tmp)
        payload = root / "payload.json"
        out = root / "out"
        payload.write_text(json.dumps(PAYLOAD, ensure_ascii=False, indent=2), encoding="utf-8")
        run([sys.executable, "-m", "fuzzyxai.cli", "list-adapters"], root)
        run([sys.executable, "-m", "fuzzyxai.cli", "list-operators"], root)
        run([sys.executable, "-m", "fuzzyxai.cli", "validate", "--payload", str(payload), "--schema", "classification"], root)
        run([sys.executable, "-m", "fuzzyxai.cli", "run", "--payload", str(payload), "--adapter", "tabular_classification", "--out", str(out)], root)
        run([sys.executable, "-m", "fuzzyxai.cli", "verify", "--route", str(out / "route.json"), "--proof", str(out / "proof_trace.json")], root)
        run([sys.executable, "-m", "fuzzyxai.cli", "render", "--route", str(out / "route.json"), "--out", str(out / "cli_dashboard.png")], root)
        run([sys.executable, "-m", "fuzzyxai.cli", "package", "--route", str(out / "route.json"), "--out", str(root / "cli_package.zip")], root)
        for path in [out / "route.json", out / "proof_trace.json", out / "operator_dashboard.png", out / "cli_dashboard.png", root / "cli_package.zip"]:
            if not path.exists() or path.stat().st_size == 0:
                raise SystemExit(f"missing CLI output: {path}")
    print("fuzzyxai-cli-check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
