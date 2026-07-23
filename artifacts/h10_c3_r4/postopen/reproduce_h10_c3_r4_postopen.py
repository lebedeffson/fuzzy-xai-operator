#!/usr/bin/env python3
"""Reproduce H10-C3 R4 after the sealed seed has been disclosed.

This verifier never calls the official opening/scoring API and therefore does
not create or modify an opening record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from h10_c3_r4 import runner
from h10_c3_r4.scientific_classifier import (
    classify_confirmatory_result,
)
from h10_c3_r4.secure_sealed import (
    STUDY_ID,
    _cases_from_plaintext,
    _decrypt,
    verify_secure_protocol_lock,
)


PREOPEN_TAG = "h10-c3-r4-v23.3-preopen-final"
PREOPEN_COMMIT = "2e530ae132b6293ba4aa0a265cbccdc9c49bb418"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized_statistics(
    values: list[dict[str, object]],
) -> list[dict[str, object]]:
    normalized = []
    for value in values:
        item = dict(value)
        item.pop("status", None)
        normalized.append(item)
    return normalized


def _verify_tag(repo_root: Path, allowed_signers: Path) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            f"gpg.ssh.allowedSignersFile={allowed_signers}",
            "verify-tag",
            PREOPEN_TAG,
        ],
        cwd=repo_root,
        check=True,
    )
    head = subprocess.check_output(
        ["git", "rev-parse", f"{PREOPEN_TAG}^{{}}"],
        cwd=repo_root,
        text=True,
    ).strip()
    if head != PREOPEN_COMMIT:
        raise ValueError("preopen tag does not resolve to the locked commit")


def reproduce(
    repo_root: Path,
    official_dir: Path,
    output_dir: Path,
) -> Path:
    artifact_root = repo_root / "artifacts" / "h10_c3_r4"
    allowed_signers = official_dir / "PREOPEN_ALLOWED_SIGNERS"
    _verify_tag(repo_root, allowed_signers)
    verify_secure_protocol_lock(repo_root, artifact_root)

    commitment = _read_json(
        artifact_root
        / "secure_sealed"
        / "sealed_bank_commitment.json"
    )
    if not isinstance(commitment, dict):
        raise TypeError("sealed commitment must be an object")
    seed = (official_dir / "disclosed_sealed_seed.bin").read_bytes()
    if len(seed) != 32:
        raise ValueError("disclosed sealed seed must contain 32 bytes")
    seed_commitment = hashlib.sha256(
        STUDY_ID.encode()
        + str(commitment["protocol_lock_sha256"]).encode()
        + seed
    ).hexdigest()
    if seed_commitment != commitment["seed_commitment_sha256"]:
        raise ValueError("disclosed seed commitment mismatch")

    ciphertext_path = (
        artifact_root / "secure_sealed" / "sealed_ciphertext.bin"
    )
    if _sha256(ciphertext_path) != commitment[
        "encrypted_payload_sha256"
    ]:
        raise ValueError("encrypted payload checksum mismatch")
    plaintext = _decrypt(
        ciphertext_path.read_bytes(),
        seed,
        str(commitment["protocol_lock_sha256"]),
    )
    if hashlib.sha256(plaintext).hexdigest() != commitment[
        "plaintext_commitment_sha256"
    ]:
        raise ValueError("plaintext commitment mismatch")

    payload = json.loads(plaintext)
    private_hashes = {
        item["canonical_hash"] for item in payload["templates"]
    }
    public_hashes: set[str] = set()
    for split in ("development", "protocol_validation"):
        manifest = _read_json(
            repo_root
            / "experiments"
            / "h10_c3_r4"
            / "templates"
            / split
            / "manifest.json"
        )
        if not isinstance(manifest, dict):
            raise TypeError("template manifest must be an object")
        public_hashes.update(manifest["canonical_hashes"])
    overlap = private_hashes & public_hashes
    if overlap:
        raise ValueError("private templates overlap public template banks")

    cases = _cases_from_plaintext(plaintext)
    rows = [
        row for case in cases for row in runner._run_case(case)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    reproduced_csv = output_dir / "reproduced_sealed.csv"
    with reproduced_csv.open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    with (official_dir / "sealed.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        official_rows = list(csv.DictReader(stream))
    with reproduced_csv.open(
        encoding="utf-8", newline=""
    ) as stream:
        reproduced_rows = list(csv.DictReader(stream))
    if len(official_rows) != len(reproduced_rows):
        raise ValueError("reproduced row count differs")
    mismatch_count = 0
    for official, reproduced in zip(
        official_rows, reproduced_rows, strict=True
    ):
        mismatch_count += sum(
            official[key] != reproduced[key]
            for key in official
            if key != "runtime_ms"
        )
    if mismatch_count:
        raise ValueError(
            f"{mismatch_count} non-runtime result fields differ"
        )

    reproduced_statistics_path = runner.analyze(
        "sealed_reproduction",
        rows,
    )
    reproduced_statistics = _read_json(
        reproduced_statistics_path
    )
    official_statistics = _read_json(
        official_dir / "sealed_statistics.json"
    )
    if not isinstance(reproduced_statistics, list) or not isinstance(
        official_statistics, list
    ):
        raise TypeError("statistics files must contain arrays")
    if _normalized_statistics(
        reproduced_statistics
    ) != _normalized_statistics(official_statistics):
        raise ValueError("registered statistics did not reproduce")

    safety = runner._sealed_safety(rows)
    classification = classify_confirmatory_result(
        official_statistics,
        **safety,
    )
    official_status = _read_json(official_dir / "sealed_status.json")
    if not isinstance(official_status, dict):
        raise TypeError("official sealed status must be an object")
    for key in ("H10-C3a", "H10-C3b", "scientific_status"):
        if classification[key] != official_status[key]:
            raise ValueError(f"classification mismatch for {key}")
    if (
        artifact_root / "secure_sealed" / "opening_record.json"
    ).exists():
        raise PermissionError(
            "postopen reproduction must not create an opening record"
        )

    report = {
        "reproduced_at_utc": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "source_tag": PREOPEN_TAG,
        "source_commit": PREOPEN_COMMIT,
        "signed_tag_verification": "PASS",
        "disclosed_seed_commitment": "PASS",
        "encrypted_payload_sha256": "PASS",
        "plaintext_commitment_sha256": "PASS",
        "private_template_count": len(private_hashes),
        "public_template_overlap": len(overlap),
        "case_count": len(cases),
        "method_row_count": len(rows),
        "non_runtime_row_reproduction": "PASS",
        "non_runtime_mismatch_count": mismatch_count,
        "statistics_reproduced": "PASS",
        "classification_reproduced": "PASS",
        "reproduced_classification": classification,
        "official_opening_record_untouched": True,
        "reproduction_opening_record_created": False,
        "official_results_sha256": _sha256(
            official_dir / "sealed.csv"
        ),
        "reproduced_results_sha256": _sha256(reproduced_csv),
        "note": (
            "Runtime is remeasured and need not be byte-identical; "
            "all scientific fields, statistics, and classifications "
            "must reproduce."
        ),
    }
    output = output_dir / "independent_reproduction.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = reproduce(
        args.repo.resolve(),
        args.official_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(output)


if __name__ == "__main__":
    main()
