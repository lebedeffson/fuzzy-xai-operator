#!/usr/bin/env python3
"""Build and verify the shareable blind-analysis input archive."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from fuzzyxai.ai_pre_review.contracts import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    output_dir = ROOT / "release_artifacts/ai_pre_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"fuzzyxai-ai-pre-review-analysis-bundle-{commit[:12]}.zip"
    study = ROOT / "study/ai_pre_review"
    with tempfile.TemporaryDirectory() as temp:
        bundle = Path(temp) / "fuzzyxai-ai-pre-review-analysis"
        bundle.mkdir()
        for source, target in (
            (study / "README.md", bundle / "README.md"),
            (study / "rubric_v1.yaml", bundle / "rubric_v1.yaml"),
            (study / "ai_review_schema.json", bundle / "ai_review_schema.json"),
            (study / "master_explanation_log.jsonl", bundle / "master_explanation_log.jsonl"),
            (study / "AI_REVIEW_MASTER_LOG.md", bundle / "AI_REVIEW_MASTER_LOG.md"),
            (study / "batch_manifest.json", bundle / "batch_manifest.json"),
            (study / "source_case_evidence.jsonl", bundle / "evidence_snapshots" / "source_case_evidence.jsonl"),
            (study / "method_identity_key.encrypted", bundle / "method_identity_key.encrypted"),
        ):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        shutil.copytree(study / "chatgpt_batches", bundle / "chatgpt_batches")
        lock = study / "confirmatory_protocol_lock.json"
        if lock.is_file():
            shutil.copy2(lock, bundle / "protocol_lock.json")
        else:
            (bundle / "protocol_lock.json").write_text(
                json.dumps({"status": "not_locked", "is_protocol_lock": False, "reason": "formative AI review has not been run"}, indent=2) + "\n",
                encoding="utf-8",
            )
        _case_index(bundle)
        files = sorted(path for path in bundle.rglob("*") if path.is_file())
        checksums = "".join(f"{sha256_file(path)}  {path.relative_to(bundle).as_posix()}\n" for path in files)
        (bundle / "SHA256SUMS").write_text(checksums, encoding="utf-8")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    name = (Path(bundle.name) / path.relative_to(bundle)).as_posix()
                    info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    handle.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    _verify(archive)
    digest = sha256_file(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"PASS: ai_pre_review_archive path={archive.relative_to(ROOT)} sha256={digest}")


def _case_index(bundle: Path) -> None:
    import csv
    rows = [json.loads(line) for line in (bundle / "evidence_snapshots/source_case_evidence.jsonl").read_text(encoding="utf-8").splitlines()]
    with (bundle / "case_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "object_id_hash", "modality", "split", "stratum"])
        for row in rows:
            writer.writerow([row["case_id"], row["object_id_hash"], row["modality"], row["split"], "|".join(row["stratum"])])


def _verify(archive: Path) -> None:
    with tempfile.TemporaryDirectory() as temp:
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(temp)
        root = Path(temp) / "fuzzyxai-ai-pre-review-analysis"
        required = {"README.md", "protocol_lock.json", "rubric_v1.yaml", "master_explanation_log.jsonl", "batch_manifest.json", "method_identity_key.encrypted", "SHA256SUMS"}
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise RuntimeError(f"archive missing files: {missing}")
        for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            if sha256_file(root / relative) != expected:
                raise RuntimeError(f"archive checksum mismatch: {relative}")


if __name__ == "__main__":
    main()
