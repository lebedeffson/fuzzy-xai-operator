#!/usr/bin/env python3
"""Build a deterministic technical prelock bundle from verified artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile

from common import EVIDENCE, ROOT, STUDY, load, sha256


def main() -> None:
    manifest = load(EVIDENCE / "prelock_manifest.json")
    if manifest.get("status") != "pass" or manifest.get("confirmatory_claim_allowed") is not False:
        raise SystemExit("FAIL: verified prelock manifest is required")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = [ROOT / row["path"] for row in manifest["artifacts"]]
    files.extend(
        (
            EVIDENCE / "prelock_manifest.json",
            STUDY / "ai_formative_run2/fuzzyxai-ai-formative-run2-input.zip.sha256",
        )
    )
    output_dir = ROOT / "release_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"fuzzyxai-final-confirmatory-prelock-{head[:12]}.zip"
    checksums = "".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in sorted(files))
    boundary = {
        "source_commit": head,
        "scope": "technical_prelock_formative",
        "confirmatory_test_opened": False,
        "confirmatory_claim_allowed": False,
        "stable_release_allowed": False,
        "remaining_external_inputs": [
            "five independent sealed dataset packages",
            "isolated OOF and test split manifest",
            "encrypted test-label vaults",
            "clean-session 240-case AI formative run 2 scores",
        ],
    }
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(files):
            _write(bundle, path.relative_to(ROOT).as_posix(), path.read_bytes())
        _write(bundle, "SHA256SUMS", checksums.encode())
        _write(bundle, "BOUNDARY.json", (json.dumps(boundary, indent=2, sort_keys=True) + "\n").encode())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = archive.with_suffix(".zip.sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"PASS: final_prelock_bundle path={archive.relative_to(ROOT)} sha256={digest}")


def _write(bundle: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    bundle.writestr(info, payload)


if __name__ == "__main__":
    main()
