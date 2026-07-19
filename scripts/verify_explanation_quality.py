from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "release_evidence/explanation_quality"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_checksums(directory: Path) -> None:
    for line in (directory / "checksums.sha256").read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = directory / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"checksum mismatch: {path}")


def verify(evidence: Path, *, require_all_pass: bool = False) -> dict[str, Any]:
    _verify_checksums(evidence)
    report = _load(evidence / "explanation_quality_report.json")
    families = report.get("families", [])
    if len(families) != report.get("family_count"):
        raise ValueError("quality family count does not match report summary")
    threshold = float(report.get("stability_threshold", 0.67))
    weak_stability = [
        row["configuration_id"]
        for row in families
        if row.get("stability") is not None and float(row["stability"]) < threshold
    ]
    if weak_stability:
        raise ValueError(f"top-reason stability is below {threshold}: {weak_stability}")
    missing_provenance = [
        row["configuration_id"]
        for row in families
        if row.get("status") == "pass" and row.get("provenance_complete") is not True
    ]
    if missing_provenance:
        raise ValueError(f"verified explanations lack provenance: {missing_provenance}")
    incomplete_user_text = [
        row["configuration_id"]
        for row in families
        if row.get("status") == "pass" and row.get("user_explanation_complete") is not True
    ]
    if incomplete_user_text:
        raise ValueError(f"verified explanations fail human-layer checks: {incomplete_user_text}")
    if require_all_pass:
        failed = [row["configuration_id"] for row in families if row.get("status") != "pass"]
        if failed:
            raise ValueError(f"quality report contains non-pass configurations: {failed}")
        if int(report.get("pass_count", -1)) != len(families):
            raise ValueError("quality pass count is incomplete")
    return {
        "family_count": len(families),
        "pass_count": int(report.get("pass_count", 0)),
        "measured_stability_count": sum(row.get("stability") is not None for row in families),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify aggregate FuzzyXAI explanation-quality evidence.")
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--require-all-pass", action="store_true")
    args = parser.parse_args()
    result = verify(args.evidence, require_all_pass=args.require_all_pass)
    print(f"PASS: explanation_quality {result['pass_count']}/{result['family_count']}")
    print(f"PASS: measured_stability {result['measured_stability_count']}")


if __name__ == "__main__":
    main()
