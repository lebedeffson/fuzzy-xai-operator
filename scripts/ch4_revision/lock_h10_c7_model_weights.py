#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from fuzzyxai.experiments.h10_c7_model_lock import (
    snapshot_path,
    snapshot_record,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method-commit", required=True)
    parser.add_argument("--make-read-only", action="store_true")
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for group in ("dense_encoders", "cross_encoders"):
        for item in registry.get(group, []):
            record = snapshot_record(
                snapshot_path(
                    args.cache_root,
                    str(item["model_name"]),
                    str(item["revision"]),
                ),
                item,
            )
            weight_digest = str(record["weights_sha256"])
            records.append(record)
            if item.get("weights_sha256") not in {
                "PENDING_LOCAL_WEIGHT_VERIFICATION",
                weight_digest,
            }:
                raise SystemExit(
                    f"registered weight digest changed for {item['model_name']}"
                )
            item["weights_sha256"] = weight_digest

    lock = {
        "method_commit": args.method_commit,
        "network_allowed_during_scoring": False,
        "local_files_only": True,
        "model_count": len(records),
        "models": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(lock, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    args.registry.write_text(
        json.dumps(registry, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.make_read_only:
        for record in records:
            snapshot = Path(record["local_path"])
            for path in snapshot.rglob("*"):
                if path.is_file():
                    os.chmod(path.resolve(), 0o444)
            for path in sorted(
                (item for item in snapshot.rglob("*") if item.is_dir()),
                reverse=True,
            ):
                os.chmod(path, 0o555)
            os.chmod(snapshot, 0o555)
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
