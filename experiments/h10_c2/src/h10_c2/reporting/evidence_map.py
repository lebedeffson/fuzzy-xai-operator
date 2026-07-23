from __future__ import annotations

from ..hashing import file_sha256, read_json, write_json
from ..paths import ARTIFACT_ROOT


def build_evidence_map() -> list[dict]:
    entries = []
    design_path = ARTIFACT_ROOT / "power" / "recommended_design.json"
    design = read_json(design_path)
    for claim in ("h10_c2a", "h10_c2b"):
        for metric, value in design[claim].items():
            if isinstance(value, (int, float)):
                entries.append(
                    {
                        "claim_id": claim.upper().replace("_", "-"),
                        "metric": metric,
                        "value": value,
                        "source_file": str(design_path.relative_to(ARTIFACT_ROOT)),
                        "locator": f"{claim}.{metric}",
                        "sha256": file_sha256(design_path),
                        "status": "design_not_confirmatory",
                    }
                )
    write_json(ARTIFACT_ROOT / "audit" / "evidence_map.json", entries)
    return entries

