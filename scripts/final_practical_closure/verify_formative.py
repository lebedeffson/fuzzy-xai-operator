#!/usr/bin/env python3
"""Verify practical formative evidence without promoting its claims."""

from __future__ import annotations

import json

from common import EXPERIMENTS, FORMATIVE, ROOT, load_json, sha256, verify_immutable_results, verify_sha256s, write_json


def main() -> None:
    verify_immutable_results()
    summary = load_json(FORMATIVE / "summary.json")
    if summary.get("confirmatory_test_opened") is not False or summary.get("confirmatory_claim_allowed") is not False:
        raise SystemExit("FAIL: formative summary crosses the confirmatory boundary")
    files: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        directory = FORMATIVE / experiment
        count = verify_sha256s(directory)
        if count < 10:
            raise SystemExit(f"FAIL: incomplete evidence package for {experiment}")
        payload = load_json(directory / "summary.json")
        claim = load_json(directory / "claim_status.json")
        protocol = load_json(directory / "protocol.json")
        split = load_json(directory / "split_manifest.json")
        if payload.get("confirmatory_claim_allowed") is not False:
            raise SystemExit(f"FAIL: {experiment} permits a confirmatory claim")
        if claim != {"claim_allowed": False, "status": "formative_only"}:
            raise SystemExit(f"FAIL: invalid formative claim boundary for {experiment}")
        if protocol.get("confirmatory_test_opened") is not False or split.get("test_opened") is not False:
            raise SystemExit(f"FAIL: {experiment} reports opened test data")
        _verify_parquet(directory / "raw_results.parquet")
        for path in sorted(directory.iterdir()):
            if path.is_file():
                files.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "size": path.stat().st_size})
    _verify_h3_partition()
    manifest = {
        "schema_version": "1.0",
        "phase": "formative_development",
        "confirmatory_test_opened": False,
        "confirmatory_claim_allowed": False,
        "experiment_count": len(EXPERIMENTS),
        "files": files,
    }
    write_json(FORMATIVE / "manifest.json", manifest)
    print(f"PASS: practical_formative_evidence experiments={len(EXPERIMENTS)} files={len(files)} confirmatory_opened=false")


def _verify_parquet(path) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit("FAIL: pyarrow is required to verify raw evidence") from error
    table = pq.read_table(path)
    if table.num_rows < 1:
        raise SystemExit(f"FAIL: empty parquet evidence {path.relative_to(ROOT)}")


def _verify_h3_partition() -> None:
    payload = load_json(FORMATIVE / "H3_practical/summary.json")
    for row in payload.get("matched_budget_rows", []):
        if not isinstance(row, dict) or row.get("policy") == "always_review":
            continue
        total = float(row["automatic_coverage"]) + float(row["review_rate"]) + float(row["block_rate"])
        if abs(total - 1.0) > 1e-9:
            raise SystemExit(f"FAIL: H3 action partition is incomplete for {row.get('policy')}")
    raw = FORMATIVE / "H3_practical/raw_results.jsonl"
    for line in raw.read_text(encoding="utf-8").splitlines():
        json.loads(line)


if __name__ == "__main__":
    main()

