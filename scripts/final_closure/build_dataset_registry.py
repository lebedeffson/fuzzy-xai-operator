#!/usr/bin/env python3
"""Build candidate registry and reject any known formative identity reuse."""

from __future__ import annotations

import json

from common import ROOT, STUDY, write


CANDIDATES = (
    ("bank_marketing", "tabular", "UCI Bank Marketing", "CC BY 4.0"),
    ("statlog_german_credit", "tabular", "UCI Statlog German Credit", "CC BY 4.0"),
    ("cifar10", "image", "CIFAR-10", "dataset-specific research terms"),
    ("ag_news", "text", "AG News", "dataset-specific research terms"),
    ("ford_a", "timeseries", "UCR FordA", "UCR archive terms"),
)


def main() -> None:
    formative_ids, formative_hashes = _formative_identities()
    rows = []
    for dataset_id, modality, source, license_name in CANDIDATES:
        rows.append(
            {
                "dataset_id": dataset_id,
                "modality": modality,
                "source": source,
                "license": license_name,
                "status": "candidate_not_downloaded",
                "formative_id_overlap": dataset_id in formative_ids,
                "download_sha256": None,
                "preprocessing_sha256": None,
                "label_vault_sha256": None,
            }
        )
    blockers = [row["dataset_id"] for row in rows if row["formative_id_overlap"]]
    write(
        STUDY / "confirmatory_dataset_registry.json",
        {
            "schema_version": "1.0",
            "datasets": rows,
            "known_formative_dataset_ids": sorted(formative_ids),
            "known_formative_hashes": sorted(formative_hashes),
            "known_formative_hash_count": len(formative_hashes),
            "status": "blocked_pending_download_and_sealing" if not blockers else "blocked_formative_overlap",
            "blockers": blockers,
        },
    )
    print(f"PASS: final_dataset_registry candidates={len(rows)} formative_overlap={len(blockers)} sealed=0")


def _formative_identities():
    ids, hashes = set(), set()
    for path in ROOT.rglob("*manifest*.json"):
        if any(part in {".git", ".venv", "release_artifacts", "final_confirmatory_closure"} for part in path.parts):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        text = json.dumps(payload, sort_keys=True)
        for key in ("dataset_id", "dataset", "dataset_name"):
            _collect(payload, key, ids)
        import re

        hashes.update(re.findall(r"\b[0-9a-f]{64}\b", text))
    return ids, hashes


def _collect(value, key, output):
    if isinstance(value, dict):
        if isinstance(value.get(key), str):
            output.add(value[key])
        for child in value.values():
            _collect(child, key, output)
    elif isinstance(value, list):
        for child in value:
            _collect(child, key, output)


if __name__ == "__main__":
    main()
