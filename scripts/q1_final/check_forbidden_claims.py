#!/usr/bin/env python3
"""Block predictive-safety and unsupported universal wording in current public material."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGETS = (
    ROOT / "README.md",
    ROOT / "RELEASE_NOTES.md",
    ROOT / "RELEASE_STATUS.md",
    ROOT / "docs/reproduction",
    ROOT / "reports/q1_final",
    ROOT / "release_evidence/q1_final",
    ROOT / "framework/fuzzyxai/fuzzyxai",
)
FORBIDDEN = (
    "critical rupture predicts error",
    "critical rupture predicts safety",
    "predictive safety gain",
    "safety predictor",
    "proof of unsafe prediction",
    "guarantees safe action",
    "понятно всем пользователям",
    "гарантирует безопасное действие",
)
DECLARATIVE_EXEMPTIONS = ("forbidden", "prohibited", "запрещ", "must not", "does not")


def files() -> list[Path]:
    rows = []
    for target in TARGETS:
        if target.is_file():
            rows.append(target)
        elif target.is_dir():
            rows.extend(path for path in target.rglob("*") if path.is_file() and path.suffix.lower() in {".py", ".md", ".json", ".yaml", ".yml"})
    return sorted(set(rows))


def main() -> None:
    findings = []
    for path in files():
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            lowered = line.lower()
            if any(term in lowered for term in FORBIDDEN) and not any(term in lowered for term in DECLARATIVE_EXEMPTIONS):
                findings.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    if findings:
        raise RuntimeError("forbidden current claims:\n" + "\n".join(findings))
    output = ROOT / "release_evidence/q1_final/forbidden_claims_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "PASS",
                "files_scanned": len(files()),
                "finding_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"PASS: q1_final_forbidden_claims files={len(files())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main()
