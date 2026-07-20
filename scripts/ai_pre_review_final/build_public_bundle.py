#!/usr/bin/env python3
"""Build a deterministic public reviewer bundle without the hidden scoring key."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from fuzzyxai.ai_pre_review_final.contracts import read_jsonl, sha256_file

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    output_dir = ROOT / "release_artifacts/ai_pre_review_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"fuzzyxai-ai-pre-review-final-bundles-{commit[:12]}.zip"
    study = ROOT / "study/ai_pre_review_final"
    with tempfile.TemporaryDirectory() as temp:
        bundle = Path(temp) / "fuzzyxai-ai-pre-review-final-bundles"
        bundle.mkdir()
        (bundle / "README.md").write_text(
            "# Blind explanation review inputs\n\n"
            "This package contains reviewer-visible evidence only. True outcomes, expected actions, method identities, original strata and answer-key annotations are excluded.\n\n"
            "Status: formative input ready. No AI or human review result is included. Confirmatory use is forbidden until real formative acceptance and protocol lock.\n",
            encoding="utf-8",
        )
        public_source = study / "public_formative"
        formative_rows = read_jsonl(public_source / "reviewer_cases.jsonl")
        formative_ids = {str(row["case_id"]) for row in formative_rows}
        if len(formative_ids) != 240 or len(formative_rows) != 720:
            raise RuntimeError("public formative boundary must contain 240 cases and 720 variants")
        formative_batches = _write_batches(bundle, formative_rows)
        formative_manifest = {
            "schema_version": "2.0",
            "stage": "formative",
            "case_count": len(formative_ids),
            "variant_count": len(formative_rows),
            "batches": formative_batches,
            "confirmatory_material_included": False,
            "hidden_scoring_key_included": False,
        }
        (bundle / "reviewer_cases.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in formative_rows),
            encoding="utf-8",
        )
        _master_log(bundle / "BLIND_REVIEW_MASTER_LOG.md", formative_rows)
        (bundle / "blind_batch_manifest.json").write_text(
            json.dumps(formative_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        copies = (
            (study / "reviewer_case_schema_v2.json", bundle / "reviewer_case_schema_v2.json"),
            (ROOT / "study/ai_pre_review/rubric_v1.yaml", bundle / "rubric_v1.yaml"),
            (ROOT / "study/ai_pre_review/ai_review_schema.json", bundle / "ai_review_schema.json"),
        )
        for source, target in copies:
            shutil.copy2(source, target)
        shutil.copytree(public_source / "assets", bundle / "assets")
        _case_index(formative_rows, bundle / "case_index.csv")
        public_manifest = {
            "schema_version": "2.0",
            "final_commit": commit,
            "base_commits": [
                "e34e52fb8ae62ee1be043d6d5b26a0c9214a0572",
                "bd48a9ca3795e2665e0e6a4f1ab4f4e981774c2b",
                "60ed5697d4d607df59556ea82de63527905f0f4f",
            ],
            "file_count_before_checksums": sum(path.is_file() for path in bundle.rglob("*")),
            "protocol_state": "formative_input_ready_not_reviewed",
            "claim_state": "external_claims_open",
            "external_gate_state": "OPEN_EXTERNAL",
            "hidden_scoring_key_included": False,
            "private_paths_included": False,
            "confirmatory_material_included": False,
            "reviewer_case_count": 240,
            "reviewer_variant_count": 720,
            "reviewer_cases_sha256": sha256_file(bundle / "reviewer_cases.jsonl"),
        }
        (bundle / "manifest.json").write_text(json.dumps(public_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = sorted(path for path in bundle.rglob("*") if path.is_file())
        (bundle / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(bundle).as_posix()}\n" for path in files), encoding="utf-8")
        _zip(bundle, archive)
    _verify(archive)
    digest = sha256_file(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"PASS: final_public_bundle path={archive.relative_to(ROOT)} sha256={digest}")


def _write_batches(bundle: Path, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    case_ids = sorted({str(row["case_id"]) for row in rows})
    batches: list[dict[str, object]] = []
    for number, offset in enumerate(range(0, len(case_ids), 20), 1):
        selected = set(case_ids[offset : offset + 20])
        batch_rows = [row for row in rows if str(row["case_id"]) in selected]
        path = bundle / "blind_batches/formative" / f"batch_{number:03d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in batch_rows),
            encoding="utf-8",
        )
        batches.append(
            {
                "batch_id": f"formative_batch_{number:03d}",
                "case_count": len(selected),
                "variant_count": len(batch_rows),
                "jsonl": path.relative_to(bundle).as_posix(),
                "jsonl_sha256": sha256_file(path),
            }
        )
    if len(batches) != 12:
        raise RuntimeError("formative bundle must contain 12 batches")
    return batches


def _master_log(output: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Blind Formative Review Master Log v2",
        "",
        "Reviewer-visible evidence only. Outcomes, expected actions, method identities and confirmatory cases are withheld.",
        "",
        f"Cases: {len({str(row['case_id']) for row in rows})}; variants: {len(rows)}.",
        "",
    ]
    for offset in range(0, len(rows), 60):
        lines.extend([f"## Block {offset // 60 + 1}", ""])
        for row in rows[offset : offset + 60]:
            lines.extend(
                [
                    f"### {row['case_id']} / {row['variant_id']}",
                    "",
                    "```json",
                    json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
    output.write_text("\n".join(lines), encoding="utf-8")


def _case_index(rows: list[dict[str, object]], output: Path) -> None:
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "variant_id", "modality", "task_description", "record_sha256"])
        for row in rows:
            writer.writerow([row["case_id"], row["variant_id"], row["modality"], row["task_description"], row["record_sha256"]])


def _zip(root: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            name = (Path(root.name) / path.relative_to(root)).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            handle.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _verify(archive: Path) -> None:
    with tempfile.TemporaryDirectory() as temp:
        with zipfile.ZipFile(archive) as handle:
            names = handle.namelist()
            if any("hidden_scoring_key" in name or "/private/" in name for name in names):
                raise RuntimeError("public archive contains hidden or private material")
            handle.extractall(temp)
        root = Path(temp) / "fuzzyxai-ai-pre-review-final-bundles"
        for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            if sha256_file(root / relative) != expected:
                raise RuntimeError(f"checksum mismatch: {relative}")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["hidden_scoring_key_included"] or manifest["private_paths_included"] or manifest["confirmatory_material_included"]:
            raise RuntimeError("public manifest violates private boundary")
        if any("/confirmatory/" in name for name in names):
            raise RuntimeError("public formative archive contains confirmatory material")
        rows = read_jsonl(root / "reviewer_cases.jsonl")
        if len(rows) != 720 or len({row["case_id"] for row in rows}) != 240:
            raise RuntimeError("public archive reviewer count mismatch")


if __name__ == "__main__":
    main()
