#!/usr/bin/env python3
"""Fail-closed validation of blind pre-review inputs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fuzzyxai.ai_pre_review.contracts import StudyBoundaryError, contains_method_identity, read_jsonl, sha256_file


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    study = ROOT / "study/ai_pre_review"
    manifest = json.loads((study / "batch_manifest.json").read_text(encoding="utf-8"))
    master = read_jsonl(study / "master_explanation_log.jsonl")
    if len(master) != 1080:
        raise StudyBoundaryError("master log must contain 1080 variants")
    split_cases = Counter()
    for row in master:
        split_cases[(row["split"], row["case_id"])] = 1
        if row["method_identity_encrypted"] in json.dumps({key: value for key, value in row.items() if key != "method_identity_encrypted"}):
            raise StudyBoundaryError("identity token leaked into blind content")
    if sum(split == "formative" for split, _ in split_cases) != 240 or sum(split == "confirmatory" for split, _ in split_cases) != 120:
        raise StudyBoundaryError("formative/confirmatory case boundary mismatch")
    for batch in manifest["batches"]:
        jsonl = ROOT / batch["jsonl"]
        markdown = ROOT / batch["markdown"]
        rows = read_jsonl(jsonl)
        if batch["case_count"] > 20 or batch["variant_count"] > 60:
            raise StudyBoundaryError("batch exceeds review limits")
        if sha256_file(jsonl) != batch["jsonl_sha256"] or sha256_file(markdown) != batch["markdown_sha256"]:
            raise StudyBoundaryError("batch checksum mismatch")
        if any(contains_method_identity(row) for row in rows):
            raise StudyBoundaryError("method name leaked into blind batch")
    if not (study / "method_identity_key.encrypted").is_file():
        raise StudyBoundaryError("encrypted method identity map is missing")
    print(f"PASS: ai_pre_review_inputs cases=360 variants={len(master)} batches={len(manifest['batches'])}")


if __name__ == "__main__":
    main()
