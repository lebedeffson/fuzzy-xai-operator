from __future__ import annotations


from ..hashing import write_json
from ..paths import ARTIFACT_ROOT


def run_leakage_audit() -> dict:
    public_files = list((ARTIFACT_ROOT / "data").glob("*/cases.jsonl"))
    leaks = []
    for path in public_files:
        source = path.read_text(encoding="utf-8")
        for token in ('"transactions"', '"optimal_cuts"', '"allowed_repairs"', '"clean_route"'):
            if token in source:
                leaks.append({"file": str(path), "token": token})
    report = {
        "status": "PASS" if not leaks else "BLOCKED_LEAKAGE",
        "public_files_checked": len(public_files),
        "violations": leaks,
        "sealed_opened": False,
    }
    write_json(ARTIFACT_ROOT / "audit" / "leakage_audit.json", report)
    if leaks:
        raise RuntimeError("BLOCKED_LEAKAGE")
    return report

