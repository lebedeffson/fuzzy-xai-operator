from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .common import ARTIFACT_ROOT, ROOT, load_config, sha256_file, write_json
from .compute_statistics import compute
from .run_methods import run_split


def _head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()


def run(config_path: Path) -> None:
    config = load_config(config_path)
    lock_path = ARTIFACT_ROOT / "lock" / "protocol_lock.json"
    opening_path = ARTIFACT_ROOT / "opening" / "opening_record.json"
    if not lock_path.exists():
        raise RuntimeError("sealed scoring forbidden: protocol lock is absent")
    if opening_path.exists():
        raise RuntimeError("sealed scoring forbidden: opening count would exceed one")
    lock = json.loads(lock_path.read_text())
    if _head() != lock["implementation_commit"]:
        raise RuntimeError("HEAD differs from locked implementation commit")
    if sha256_file(config_path) != lock["protocol_sha256"]:
        raise RuntimeError("protocol changed after lock")
    opening = {
        "study_id": config["study_id"],
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "opening_count": 1,
        "purpose": "single_confirmatory_scoring",
        "implementation_commit": _head(),
        "post_lock_tuning": False,
    }
    write_json(opening_path, opening)
    try:
        run_split("sealed_test", ARTIFACT_ROOT / "confirmatory")
        registry = compute(config_path)
    except Exception:
        write_json(
            ARTIFACT_ROOT / "opening" / "invalid_marker.json",
            {"status": "invalid_after_single_opening", "repeat_opening_allowed": False},
        )
        raise
    write_json(
        ARTIFACT_ROOT / "opening" / "post_scoring_audit.json",
        {
            "opening_count": 1,
            "post_lock_tuning": False,
            "labels_exported": False,
            "repeat_opening_allowed": False,
            "claim_statuses": registry["claims"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "h10_final_gold_protocol.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
