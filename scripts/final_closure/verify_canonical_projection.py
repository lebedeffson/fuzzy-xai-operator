#!/usr/bin/env python3
"""Verify exact H7-A preservation and measure the top-k projection trade-off."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from common import ROOT, STUDY, sha256, write
from oof_pipeline import DATASETS


OUTPUT = STUDY / "h7_formative"


def main() -> None:
    rows = []
    total, preserved = 0, 0
    for dataset_id in DATASETS:
        feature_path = STUDY / f"oof_features/{dataset_id}.jsonl"
        evidence_path = STUDY / f"oof_features/canonical/{dataset_id}.jsonl"
        with feature_path.open(encoding="utf-8") as features, evidence_path.open(encoding="utf-8") as evidence:
            for feature_line, evidence_line in zip(features, evidence, strict=True):
                feature = json.loads(feature_line)
                payload = json.loads(evidence_line)
                canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
                digest = hashlib.sha256(canonical.encode()).hexdigest()
                total += 1
                preserved += digest == feature["canonical_evidence_sha256"]
                values = np.abs(np.asarray([item["contribution"] for item in payload["components"]], dtype=float))
                top_count = min(5, len(values))
                retained = float(np.sum(np.sort(values)[-top_count:]) / max(np.sum(values), 1e-12))
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "object_id_hash": feature["object_id_hash"],
                        "component_count": len(values),
                        "top_k": top_count,
                        "length_reduction": 1.0 - top_count / max(1, len(values)),
                        "attribution_mass_retained": retained,
                        "canonical_hash_preserved": digest == feature["canonical_evidence_sha256"],
                        "perturbation_stability": feature["route"]["perturbation_stability"],
                    }
                )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = OUTPUT / "projection_rows.jsonl"
    raw.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "phase": "formative_real_oof",
        "H7_A": {
            "canonical_hash_preservation_rate": preserved / max(1, total),
            "target": 1.0,
            "status": "pass" if preserved == total else "fail",
        },
        "H7_B": {
            "mean_length_reduction": float(np.mean([row["length_reduction"] for row in rows])),
            "mean_attribution_mass_retained": float(np.mean([row["attribution_mass_retained"] for row in rows])),
            "mean_source_perturbation_stability": float(np.mean([row["perturbation_stability"] for row in rows])),
            "stability_gain_after_projection": None,
            "status": "blocked_projection_stability_not_yet_measured",
        },
        "objects": total,
        "raw_rows": {"path": raw.relative_to(ROOT).as_posix(), "sha256": sha256(raw)},
        "sealed_test_opened": False,
        "confirmatory_claim_allowed": False,
    }
    write(OUTPUT / "summary.json", summary)
    if preserved != total:
        raise SystemExit(f"FAIL: H7-A canonical preservation {preserved}/{total}")
    print(f"PASS: final_canonical_evidence objects={total} preservation=1.0 h7b=blocked_formative")


if __name__ == "__main__":
    main()
