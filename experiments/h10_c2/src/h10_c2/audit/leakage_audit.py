from __future__ import annotations

import json

from ..hashing import object_sha256, write_json
from ..paths import ARTIFACT_ROOT, REPO_ROOT


def run_leakage_audit() -> dict:
    public_files = list((ARTIFACT_ROOT / "data").glob("*/cases.jsonl"))
    leaks = []
    for path in public_files:
        source = path.read_text(encoding="utf-8")
        for token in ('"transactions"', '"optimal_cuts"', '"allowed_repairs"', '"clean_route"'):
            if token in source:
                leaks.append({"file": str(path), "token": token})
    old_hashes = set()
    old_root = REPO_ROOT / "artifacts" / "h10_final_gold" / "data"
    for path in old_root.glob("*_inputs.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            old_hashes.add(object_sha256(payload.get("observed_graph", payload)))
    new_hashes = set()
    for path in public_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            new_hashes.add(object_sha256(payload["observed_route"]))
    overlap = old_hashes.intersection(new_hashes)
    if overlap:
        leaks.append({"file": "case_hash_audit", "token": f"old_new_overlap:{len(overlap)}"})
    report = {
        "status": "PASS" if not leaks else "BLOCKED_LEAKAGE",
        "public_files_checked": len(public_files),
        "violations": leaks,
        "old_case_hashes_checked": len(old_hashes),
        "new_case_hashes_checked": len(new_hashes),
        "old_new_case_hash_intersection": len(overlap),
        "sealed_opened": False,
    }
    write_json(ARTIFACT_ROOT / "audit" / "leakage_audit.json", report)
    if leaks:
        raise RuntimeError("BLOCKED_LEAKAGE")
    return report
