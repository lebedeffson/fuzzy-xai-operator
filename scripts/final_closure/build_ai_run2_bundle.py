#!/usr/bin/env python3
"""Build a self-contained, leakage-free run-2 reviewer bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from common import ROOT, STUDY, sha256, write


SOURCE = ROOT / "study/ai_pre_review_final/public_formative/reviewer_cases.jsonl"
ASSETS = ROOT / "study/ai_pre_review_final/public_formative/assets"
RUBRIC = ROOT / "study/ai_pre_review/rubric_v1.yaml"
REVIEW_SCHEMA = ROOT / "study/ai_pre_review/ai_review_schema.json"
CASE_SCHEMA = ROOT / "study/ai_pre_review_final/reviewer_case_schema_v2.json"
FORBIDDEN = {
    "is_correct",
    "true_label",
    "stratum",
    "expected_action",
    "ground_truth",
    "known_contradictions",
    "known_unsupported_claims",
    "structural_rupture",
}


def forbidden_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(FORBIDDEN & set(value))
        for child in value.values():
            found.update(forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(forbidden_keys(child))
    return found


def main() -> None:
    records = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()]
    cases = sorted({record["case_id"] for record in records})
    if len(cases) != 240 or len(records) != 720:
        raise SystemExit("FAIL: run-2 source must contain 240 cases and three variants per case")
    for record in records:
        leaked = forbidden_keys(record)
        if leaked or record.get("claim_evidence_coverage") != 1.0:
            raise SystemExit(f"FAIL: blind run-2 input invalid: {sorted(leaked)}")
        if not record.get("candidate_explanation", {}).get("limitations"):
            raise SystemExit("FAIL: every run-2 card requires an explicit limitation")
    output = STUDY / "ai_formative_run2"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "fuzzyxai-ai-formative-run2-input.zip"
    with tempfile.TemporaryDirectory() as temporary:
        bundle = Path(temporary) / "fuzzyxai-ai-formative-run2-input"
        bundle.mkdir()
        input_path = bundle / "reviewer_cases.jsonl"
        input_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
            encoding="utf-8",
        )
        batches = _write_batches(bundle, cases, records)
        shutil.copy2(RUBRIC, bundle / "rubric_v1.yaml")
        shutil.copy2(REVIEW_SCHEMA, bundle / "ai_review_schema.json")
        shutil.copy2(CASE_SCHEMA, bundle / "reviewer_case_schema_v2.json")
        shutil.copytree(ASSETS, bundle / "assets")
        commit = _git_head()
        protocol = {
            "schema_version": "2.0",
            "status": "input_ready_scores_not_run",
            "source_commit": commit,
            "session_type": "temporary_clean_chat",
            "prior_context": False,
            "project_context": False,
            "memory_used": False,
            "review_status": "fully_blind",
            "run": "formative_run_2",
            "case_count": len(cases),
            "variant_count": len(records),
            "blind": True,
            "clean_session_required": True,
            "ai_review_is_external_validation": False,
            "confirmatory_material_included": False,
            "hidden_scoring_key_included": False,
            "acceptance": {
                "critical_unsupported_claims": 0,
                "critical_contradictions": 0,
                "critical_unjustified_actions": 0,
                "minimum_medians": {
                    "uncertainty_honesty": 3,
                    "clarity": 3,
                    "limitation_completeness": 3,
                },
            },
            "reviewer_input_sha256": sha256(input_path),
            "batches": batches,
        }
        write(bundle / "protocol.json", protocol)
        session_template = _session_metadata_template()
        write(bundle / "session_metadata_template.json", session_template)
        (bundle / "README.md").write_text(_readme(), encoding="utf-8")
        _write_checksums(bundle)
        _zip(bundle, archive)
        shutil.copy2(input_path, output / "reviewer_cases.jsonl")
        write(output / "protocol.json", protocol)
        write(output / "session_metadata_template.json", session_template)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    _verify_archive(archive)
    print(f"PASS: final_ai_run2_input cases=240 variants=720 scores=not_run sha256={digest}")


def _write_batches(bundle: Path, cases: list[str], records: list[dict[str, object]]) -> list[dict[str, object]]:
    batches = []
    for number, offset in enumerate(range(0, len(cases), 20), start=1):
        selected = set(cases[offset : offset + 20])
        rows = [row for row in records if row["case_id"] in selected]
        path = bundle / f"batches/batch_{number:03d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        batches.append(
            {
                "batch_id": f"formative_run2_batch_{number:03d}",
                "case_count": len(selected),
                "variant_count": len(rows),
                "path": path.relative_to(bundle).as_posix(),
                "sha256": sha256(path),
            }
        )
    return batches


def _session_metadata_template() -> dict[str, object]:
    return {
        "session_type": "temporary_clean_chat",
        "prior_context": False,
        "project_context": False,
        "memory_used": False,
        "review_status": "fully_blind",
        "run": "formative_run_2",
        "cases": 240,
        "variants": 720,
        "reviewer_model_label": "FILL_AFTER_REVIEW",
    }


def _readme() -> str:
    return """# FuzzyXAI blind formative review run 2

This archive contains reviewer-visible evidence only. It contains no true labels, correctness outcomes,
expected actions, hidden strata, method identities, confirmatory cases or scoring key.

Use a new temporary chat with no project history or memory. Upload only this ZIP and send this prompt:

> Проведи слепое формирующее рецензирование всех пакетов. Используй исключительно rubric и evidence внутри архива.
> Не пытайся определить названия методов, происхождение вариантов, правильность прогноза или ожидаемое действие.
> Верни строго валидный JSONL по приложенной схеме. Не используй сведения из других разговоров.

Review all 12 batches. Return one output row for every case/variant pair, 720 rows total. Preserve each batch ID
and input batch SHA256 from protocol.json. Use ai_run_id=AI_RUN_2 and the source_commit from protocol.json.
Save the completed session metadata as session_metadata.json beside reviews.jsonl. AI review is formative text
quality evidence, not external or domain validation.
"""


def _write_checksums(bundle: Path) -> None:
    files = sorted(path for path in bundle.rglob("*") if path.is_file())
    (bundle / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(bundle).as_posix()}\n" for path in files),
        encoding="utf-8",
    )


def _zip(bundle: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(bundle.rglob("*")):
            if not path.is_file():
                continue
            name = (Path(bundle.name) / path.relative_to(bundle)).as_posix()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            handle.writestr(info, path.read_bytes())


def _verify_archive(archive: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        if any("hidden_scoring_key" in name or "/private/" in name or "confirmatory" in name for name in names):
            raise SystemExit("FAIL: run-2 archive contains hidden or confirmatory material")
        required = {"rubric_v1.yaml", "ai_review_schema.json", "README.md", "SHA256SUMS"}
        basenames = {Path(name).name for name in names}
        if not required <= basenames or sum(name.endswith(".png") for name in names) != 60:
            raise SystemExit("FAIL: run-2 archive is incomplete")


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


if __name__ == "__main__":
    main()
