#!/usr/bin/env python3
"""Build the only external final archive from committed source and sealed evidence."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from common import EVIDENCE, ROOT, STUDY, load, sha256, write


REQUIRED = (
    STUDY / "confirmatory_protocol_lock.json",
    STUDY / "confirmatory_completion_marker.json",
    STUDY / "confirmatory/final_statistics.json",
    EVIDENCE / "final_claim_registry.json",
    ROOT / "dissertation_artifacts/final_one_zip/chapter4/Глава_4_FuzzyXAI_final.docx",
    ROOT / "dissertation_artifacts/final_one_zip/chapter4/Глава_4_FuzzyXAI_final.pdf",
)
PROHIBITED_NAMES = ("confirmatory_vault_aes256.pass", "confirmatory_label_vault.enc", ".git/", ".venv", "site/dubnaxai")


def main() -> None:
    missing = [path.relative_to(ROOT).as_posix() for path in REQUIRED if not path.is_file()]
    if missing:
        raise SystemExit(f"BLOCKED: final one ZIP prerequisites missing: {missing}")
    completion = load(STUDY / "confirmatory_completion_marker.json")
    if completion.get("status") not in {"completed_once", "completed_via_declared_scoring_recovery"}:
        raise SystemExit("BLOCKED: confirmatory run is not complete")
    head = _git("rev-parse", "HEAD")
    if _git("write-tree") != _git("rev-parse", "HEAD^{tree}"):
        raise SystemExit("BLOCKED: staged index differs from HEAD")
    output = ROOT / "release_artifacts" / f"fuzzyxai-final-practical-closure-{head[:12]}.zip"
    with tempfile.TemporaryDirectory() as temp:
        stage = Path(temp) / f"fuzzyxai-final-practical-closure-{head[:12]}"
        _populate(stage, head)
        _write_checksums(stage)
        _write_manifest(stage, head)
        _zip(stage, output)
    _verify(output)
    print(f"PASS: final_one_zip path={output.relative_to(ROOT)} sha256={sha256(output)}")


def _populate(stage: Path, head: str) -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.build_framework_release import include_in_source_release, manifest_artifact_paths

    stage.mkdir(parents=True)
    source = stage / "source"
    source.mkdir()
    with tempfile.TemporaryDirectory() as checkout_temp:
        checkout = Path(checkout_temp) / "fuzzy-xai-operator"
        checkout.mkdir()
        subprocess.run(["git", "checkout-index", "--all", f"--prefix={checkout}/"], cwd=ROOT, check=True)
        manifest_artifacts = manifest_artifact_paths(checkout)
        for path in sorted(item for item in checkout.rglob("*") if item.is_file()):
            relative = path.relative_to(checkout)
            if not include_in_source_release(relative, manifest_artifacts):
                continue
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    mappings = {
        "protocol": [
            STUDY / "protocol.json",
            STUDY / "confirmatory_protocol_lock.json",
            STUDY / "confirmatory_scoring_recovery_lock.json",
            STUDY / "confirmatory_opening_record.json",
            STUDY / "confirmatory_invalid_marker.json",
            STUDY / "confirmatory_completion_marker.json",
            STUDY / "protocol_manifest.json",
        ],
        "data_manifests": [STUDY / "confirmatory_dataset_manifest.json", STUDY / "confirmatory_split_manifest.json", STUDY / "near_duplicate_audit.json", STUDY / "final_leakage_audit.json"],
        "models": list((STUDY / "dataset_manifests").glob("*/model_manifest.json")),
        "features": [STUDY / "confirmatory_feature_manifest.json", STUDY / "p0_p1_feature_audit.json"],
        "evidence/frozen_previous": [
            ROOT / "release_evidence/final_practical_closure/claim_registry.json",
            ROOT / "release_evidence/final_practical_closure/formative/summary.json",
            EVIDENCE / "claim_status_prelock.json",
        ],
        "evidence/formative": [STUDY / "formative_real/summary.json", STUDY / "comparator_formative/summary.json", STUDY / "h7_formative/summary.json"],
        "evidence/shadow_replay": list(EVIDENCE.glob("shadow_replay*")),
        "evidence/ai_text_review": [STUDY / "ai_text_review_scope.json"],
        "evidence/ai_text_review/run2_input": [
            STUDY / "ai_formative_run2/fuzzyxai-ai-formative-run2-input.zip",
            STUDY / "ai_formative_run2/fuzzyxai-ai-formative-run2-input.zip.sha256",
            STUDY / "ai_formative_run2/protocol.json",
        ],
        "statistics": [STUDY / "confirmatory/final_statistics.json"],
        "tables": list((ROOT / "dissertation_artifacts/final_one_zip/chapter4/tables").glob("*")),
        "figures": list((ROOT / "dissertation_artifacts/final_one_zip/chapter4/figures").glob("*")),
        "chapter4": list((ROOT / "dissertation_artifacts/final_one_zip/chapter4").glob("*.*")),
        "release_status": [ROOT / "PROJECT_MEMORY.md", ROOT / "RELEASE_STATUS.md", EVIDENCE / "final_claim_registry.json"],
        "logs": [
            STUDY / "confirmatory_opening_record.json",
            STUDY / "confirmatory_invalid_marker.json",
            STUDY / "confirmatory_completion_marker.json",
        ],
    }
    for destination, paths in mappings.items():
        target = stage / destination
        target.mkdir(parents=True, exist_ok=True)
        for path in paths:
            if path.is_file() and not any(name in path.as_posix() for name in PROHIBITED_NAMES):
                shutil.copy2(path, target / path.name)
    _copy_tree(STUDY / "confirmatory/models", stage / "models/confirmatory")
    _copy_tree(STUDY / "confirmatory/features", stage / "features/sealed_test")
    _copy_tree(STUDY / "confirmatory/canonical", stage / "evidence/confirmatory/canonical")
    _copy_tree(
        STUDY / "confirmatory",
        stage / "evidence/confirmatory/results",
        exclude_directories={"models", "features", "canonical"},
    )
    (stage / "LICENSES").mkdir()
    for license_path in ROOT.glob("data/confirmatory/*/manifests/license.txt"):
        shutil.copy2(license_path, stage / "LICENSES" / f"{license_path.parents[1].name}.txt")
    (stage / "reproducibility").mkdir()
    shutil.copy2(ROOT / "Makefile", stage / "reproducibility/Makefile")
    original_lock = load(STUDY / "confirmatory_protocol_lock.json")
    recovery_lock = load(STUDY / "confirmatory_scoring_recovery_lock.json")
    lineage = {
        "source_commit": head,
        "protocol_commit": original_lock["source_commit"],
        "model_commit": original_lock["source_commit"],
        "feature_commit": original_lock["source_commit"],
        "confirmatory_prescore_commit": original_lock["source_commit"],
        "scoring_recovery_commit": recovery_lock["source_commit"],
        "chapter_commit": head,
        "single_commit_lineage": False,
        "lineage_deviation": "declared_scoring_only_recovery_and_conservative_post_scoring_claim_packaging",
    }
    write(stage / "artifact_lineage.json", lineage)
    _write_final_report(stage, head)
    (stage / "README_FIRST.md").write_text(
        "# FuzzyXAI final practical closure\n\nTechnical computational evidence only. Human comprehension, domain approval and expert-action claims are out of scope. The original scoring failure and declared scoring-only recovery are preserved in the protocol and evidence directories.\n",
        encoding="utf-8",
    )


def _copy_tree(source: Path, destination: Path, *, exclude_directories: set[str] | None = None) -> None:
    if not source.is_dir():
        return
    excluded = exclude_directories or set()
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source)
        if relative.parts and relative.parts[0] in excluded:
            continue
        if any(name in path.as_posix() for name in PROHIBITED_NAMES):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _write_final_report(stage: Path, head: str) -> None:
    statistics = load(STUDY / "confirmatory/final_statistics.json")
    claims = load(EVIDENCE / "final_claim_registry.json")
    protocol = load(STUDY / "protocol.json")
    completion = load(STUDY / "confirmatory_completion_marker.json")
    h3 = statistics["H3"]["P1_vs_baseline"]
    h3_summary = load(STUDY / "confirmatory/h3_h7_summary.json")["H3_P1"]
    h5 = next(row for row in statistics["H5-A"]["methods"] if row["method"] == "typed_route_validity")
    h6a = statistics["H6-A"]
    h7a = statistics["H7-A"]
    h9 = statistics["H9"]
    shadow = load(EVIDENCE / "shadow_replay_summary.json")
    ai_scope = load(STUDY / "ai_text_review_scope.json")
    claim_rows = claims.get("claims") or claims.get("hypotheses") or claims.get("new_claims", {})
    lines = [
        "# Final technical closure report",
        "",
        f"- Source commit: `{head}`",
        f"- Protocol lock: `{sha256(STUDY / 'confirmatory_protocol_lock.json')}`",
        f"- Completion status: `{completion.get('status')}`",
        f"- Confirmatory objects: `{h3['n']}`",
        f"- Primary endpoint: `{protocol['primary_endpoint']}`",
        f"- Original confirmatory status: `{load(STUDY / 'confirmatory_invalid_marker.json')['status']}`",
        f"- Scoring completion: `{completion.get('status')}`",
        "- Human comprehension, domain approval and expert-action claims: `out_of_scope`",
        "",
        "## H3",
        "",
        f"- Primary review budget: `{h3_summary['primary_review_budget']}`",
        f"- Best frozen baseline: `{h3_summary['baseline']}`",
        f"- Baseline invalid automatic actions: `{h3_summary['baseline_invalid_automatic_actions']}`",
        f"- FuzzyXAI P1 invalid automatic actions: `{h3_summary['fuzzyxai_invalid_automatic_actions']}`",
        f"- Relative reduction: `{h3['relative_invalid_action_reduction']}`",
        f"- 95% CI: `{h3['confidence_interval_95']}`",
        f"- Holm-adjusted p: `{h3['holm_adjusted_p']}`",
        "- Claim: `not_supported`",
        "",
        "## Bounded results",
        "",
        f"- H5-A controlled fault F1: `{h5['f1']}`; false certification: `{h5['false_certification']}`; source localization: `{h5['source_localization']}`.",
        f"- H6-A synthetic eligible-region detection: `{h6a['detection_rate']}`; FDR: `{h6a['false_discovery_rate']}`.",
        f"- H7-A canonical artifacts: `{h7a['artifacts']}`; preservation: `{h7a['canonical_hash_preservation_rate']}`.",
        f"- H9 operator-only maximum N: `{h9['maximum_objects']}`; end-to-end target met: `{h9['end_to_end_target_met']}`.",
        f"- Controlled shadow replay events: `{shadow['event_count']}`; confirmatory claim allowed: `{shadow['confirmatory_claim_allowed']}`.",
        f"- AI text review: `{ai_scope['status']}`; review records: `{ai_scope['review_records']}`.",
        "",
        "## Claim registry",
        "",
        "| Claim | Status |",
        "| --- | --- |",
    ]
    normalized_rows = (
        ({"claim_id": claim_id, "status": status} for claim_id, status in claim_rows.items())
        if isinstance(claim_rows, dict)
        else claim_rows
    )
    for row in normalized_rows:
        claim_id = row.get("claim_id", row.get("id", "unknown"))
        status = row.get("status", row.get("claim_status", "unknown"))
        lines.append(f"| `{claim_id}` | `{status}` |")
    lines.extend(
        [
            "",
            "## Evidence boundaries",
            "",
            "The archive preserves negative and inconclusive findings. A claim is enabled only by the machine-generated claim registry; absent external human evidence cannot be represented as human validation.",
            "",
            "Exact effect sizes, confidence intervals, adjusted p-values and units of analysis are stored in `statistics/final_statistics.json` and linked chapter artifacts.",
        ]
    )
    report = stage / "release_status/FINAL_ONE_ZIP_REPORT.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_checksums(stage: Path) -> None:
    rows = []
    for path in sorted(item for item in stage.rglob("*") if item.is_file() and item.name not in {"SHA256SUMS", "MANIFEST.json"}):
        rows.append(f"{sha256(path)}  {path.relative_to(stage).as_posix()}")
    (stage / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_manifest(stage: Path, head: str) -> None:
    files = [path for path in stage.rglob("*") if path.is_file()]
    payload = {"schema_version": "1.0", "source_commit": head, "file_count": len(files), "root": stage.name, "prohibited_files": []}
    (stage / "MANIFEST.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _zip(stage: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(item for item in stage.rglob("*") if item.is_file()):
            archive.write(path, f"{stage.name}/{path.relative_to(stage).as_posix()}")


def _verify(path: Path) -> None:
    with tempfile.TemporaryDirectory() as temp, zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        names = archive.namelist()
        if bad or any(prohibited in name for name in names for prohibited in PROHIBITED_NAMES):
            raise RuntimeError(f"final archive verification failed: bad={bad}")
        roots = {name.split("/", 1)[0] for name in names}
        if len(roots) != 1:
            raise RuntimeError("final archive must contain exactly one root directory")
        archive.extractall(temp)
        root = Path(temp) / next(iter(roots))
        for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            if sha256(root / relative) != expected:
                raise RuntimeError(f"internal checksum mismatch: {relative}")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


if __name__ == "__main__":
    main()
