#!/usr/bin/env python3
"""Validate the reviewer-visible evidence contract for every modality."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from fuzzyxai.ai_pre_review_final.contracts import FinalStudyError, read_jsonl


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "study/ai_pre_review_final/public_formative"
MODALITY_FIELDS = {
    "tabular": {"observed_value_anonymized", "reference_percentile"},
    "image": {"bounding_box", "region_id"},
    "text": {"phrase", "character_position"},
    "timeseries": {"interval_start", "interval_end", "signal_channel"},
}


def main() -> None:
    rows = read_jsonl(PUBLIC / "reviewer_cases.jsonl")
    failures: list[str] = []
    counts = Counter(str(row.get("modality")) for row in rows)
    if counts != Counter({name: 180 for name in MODALITY_FIELDS}):
        failures.append(f"unexpected modality counts: {dict(counts)}")
    for row in rows:
        modality = str(row.get("modality"))
        for item in row.get("interpretable_evidence", []):
            missing = MODALITY_FIELDS.get(modality, set()) - set(item)
            if missing:
                failures.append(f"{row.get('case_id')}/{row.get('variant_id')}: missing {sorted(missing)}")
            if not item.get("display_name") or not item.get("evidence_refs") or not item.get("limitations"):
                failures.append(f"{row.get('case_id')}/{row.get('variant_id')}: uninterpretable evidence item")
        if modality == "image":
            reference = row.get("observable_asset", {}).get("thumbnail_ref")
            if not reference or not (PUBLIC / str(reference)).is_file():
                failures.append(f"{row.get('case_id')}/{row.get('variant_id')}: missing image asset")
    if failures:
        raise FinalStudyError("evidence validation failed:\n" + "\n".join(failures[:30]))
    print(f"PASS: final_evidence records={len(rows)} modalities={len(counts)} image_assets={len(list((PUBLIC / 'assets/image').glob('*.png')))}")


if __name__ == "__main__":
    main()
