#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fuzzyxai.experiments.h10_c7 import _reject_gold
from fuzzyxai.experiments.h10_c7r import read_jsonl, sha256


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--leakage-report", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    gold_path = args.gold.resolve()
    protocol_dir = args.protocol_dir.resolve()
    authorization = args.authorization.resolve()
    if authorization.exists():
        raise ValueError("H10-C7R scoring authorization already exists")
    observable = read_jsonl(manifest)
    for index, row in enumerate(observable):
        _reject_gold(row, f"$[{index}]")
        if row.get("split") != "held_out":
            raise ValueError("authorization requires held_out records")
        if row.get("runtime_evidence_status") != "BUG_REPRODUCED_WITH_TRACE":
            raise ValueError("authorization requires complete runtime evidence")
        for field in ("graph_path", "runtime_events_path"):
            path = Path(str(row[field]))
            if not path.is_absolute():
                path = (manifest.parent / path).resolve()
            if not path.is_file():
                raise ValueError(f"missing held-out evidence: {path}")
    gold = read_jsonl(gold_path)
    observable_ids = {str(row["incident_id"]) for row in observable}
    gold_ids = {str(row["incident_id"]) for row in gold}
    repositories = {str(row["repository"]) for row in observable}
    exclusion = json.loads(
        (protocol_dir / "H10_C7R_EXCLUSION_LOCK.json").read_text(
            encoding="utf-8"
        )
    )
    overlap = sorted(
        repositories.intersection(exclusion["excluded_repositories"])
    )
    checks = {
        "minimum_40_incidents": len(observable) >= 40,
        "minimum_12_repositories": len(repositories) >= 12,
        "observable_gold_ids_match": observable_ids == gold_ids,
        "repository_exclusion_overlap_zero": not overlap,
        "gold_fields_absent_from_observable": True,
        "runtime_evidence_complete": all(
            row["runtime_evidence_status"] == "BUG_REPRODUCED_WITH_TRACE"
            for row in observable
        ),
    }
    leakage_report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "excluded_repository_overlap": overlap,
        "gold_leakage": 0 if all(checks.values()) else 1,
    }
    _write_json(args.leakage_report.resolve(), leakage_report)
    if not all(checks.values()):
        raise ValueError("H10-C7R authorization checks failed")
    lock_manifest = protocol_dir / "H10_C7R_LOCK_MANIFEST.json"
    payload = {
        "authorization_status": "H10_C7R_SCORING_AUTHORIZED_ONCE",
        "gold_sha256": sha256(gold_path),
        "held_out_incidents": len(observable),
        "held_out_manifest_sha256": sha256(manifest),
        "held_out_repositories": len(repositories),
        "lock_manifest_sha256": sha256(lock_manifest),
        "opening_count_before_scoring": 0,
        "protocol_id": "H10-C7R-v1",
        "single_official_scoring": True,
    }
    _write_json(authorization, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
