from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .common import ARTIFACT_ROOT, ROOT, load_config, sha256_file, write_json
from .power_analysis import analyze as analyze_power
from .validate_adjudication import validate


ALGORITHM_PATHS = (
    "gold_oracle",
    "framework/fuzzyxai/fuzzyxai/audit_h10/gold_benchmark.py",
    "baselines/h10_gold",
    "experiments/h10_gold",
    "config/h10_final_gold_protocol.yaml",
)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def freeze(config_path: Path) -> dict:
    config = load_config(config_path)
    if (ARTIFACT_ROOT / "opening" / "opening_record.json").exists():
        raise RuntimeError("protocol cannot be changed after sealed opening")
    adjudication = validate(config_path)
    if adjudication["status"] != "PASS":
        raise RuntimeError(f"manual adjudication gate failed: {adjudication['status']}")
    power = analyze_power(config_path)
    if power["status"] != "PASS":
        raise RuntimeError("power gate failed: primary development effect does not justify sealed scoring")
    dirty = _git("status", "--porcelain", "--", *ALGORITHM_PATHS)
    if dirty:
        raise RuntimeError("commit Gold algorithms and protocol before lock")
    files = [
        path for item in ALGORITHM_PATHS
        for path in ((ROOT / item).rglob("*.py") if (ROOT / item).is_dir() else (ROOT / item,))
        if path.is_file()
    ]
    lock = {
        "study_id": config["study_id"],
        "status": "LOCKED_PENDING_SINGLE_SEALED_OPENING",
        "implementation_commit": _git("rev-parse", "HEAD"),
        "protocol_sha256": sha256_file(config_path),
        "algorithm_files": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in sorted(files)
        },
        "primary_population": config["primary_population"],
        "primary_metrics": config["primary_metrics"],
        "registered_margins": config["registered_margins"],
        "safety_constraints": config["safety_constraints"],
        "bootstrap": config["bootstrap"],
        "adjudication_sha256": sha256_file(ARTIFACT_ROOT / "adjudication" / "adjudication_validation.json"),
        "power_analysis_sha256": sha256_file(ARTIFACT_ROOT / "exploratory" / "power_analysis.json"),
        "best_baseline_selected_on_development": power["best_baseline_selected_on_development"],
        "sealed_input_sha256": sha256_file(ARTIFACT_ROOT / "data" / "sealed_test_inputs.jsonl"),
        "sealed_truth_sha256": sha256_file(ROOT / ".h10_final_gold_private" / "sealed_test_truth.jsonl"),
        "opening_limit": 1,
    }
    lock["lock_sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True).encode("utf-8")).hexdigest()
    write_json(ARTIFACT_ROOT / "lock" / "protocol_lock.json", lock)
    return lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "h10_final_gold_protocol.yaml")
    args = parser.parse_args()
    freeze(args.config)


if __name__ == "__main__":
    main()
